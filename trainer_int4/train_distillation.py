import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset
from bitsandbytes.nn import Linear4bit
import deepspeed

warnings.filterwarnings('ignore')

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def distillation_loss_fn(student_logits, teacher_logits, temperature=1.0):
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

def train_epoch(epoch, wandb, alpha=0.5, temperature=1.0):
    start_time = time.time()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    if teacher_model is not None:
        teacher_model.eval()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, 
                    args.epochs * iter_per_epoch, 
                    args.learning_rate)
                    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # 学生模型前向传播
            res = model(X)
            student_logits = res.logits
            
            # 教师模型前向传播
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_res = teacher_model(X)
                    teacher_logits = teacher_res.logits

            # 计算损失
            ce_loss = loss_fct(
                student_logits.view(-1, student_logits.size(-1)), 
                Y.view(-1)
            ).view(Y.size())
            ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()
            
            if teacher_model is not None:
                distill_loss = distillation_loss_fn(student_logits, teacher_logits, temperature)
                loss = alpha * ce_loss + (1 - alpha) * distill_loss
            else:
                loss = ce_loss
                
            loss = loss / args.accumulation_steps

        # DeepSpeed梯度处理
        model.backward(loss)
        model.step()

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            log_data = {
                'epoch': epoch,
                'step': step,
                'loss': loss.item() * args.accumulation_steps,
                'lr': optimizer.param_groups[-1]['lr'],
                'time': spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
            }
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iter_per_epoch}) loss:{log_data["loss"]:.4f} lr:{log_data["lr"]:.12f}')
            
            if wandb and (not ddp or dist.get_rank() == 0):
                wandb.log(log_data)

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/distilled_{lm_config_student.hidden_size}{moe_suffix}.pth'
            
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            model.train()

def init_model(lm_config, is_teacher=False):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    
    # 初始化模型
    model = MiniMindForCausalLM(lm_config)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            model._modules[name] = Linear4bit(
                module.in_features,
                module.out_features,
                bias=False,
                quant_type="nf4",
                compute_dtype=torch.float16
            )
    
    # 加载预训练权重
    suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{suffix}.pth'
    if os.path.exists(ckp):
        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(args.device)
    Logger(f'{"教师" if is_teacher else "学生"}模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
    return model, tokenizer if not is_teacher else model

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind 知识蒸馏")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--data_path", type=str, default="../dataset/sft_data.jsonl")
    parser.add_argument("--teacher_size", type=int, default=768)
    parser.add_argument("--student_size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.7, help="CE损失权重")
    parser.add_argument("--temperature", type=float, default=2.0, help="蒸馏温度")

    args = parser.parse_args()
    
    # 配置模型
    lm_config_teacher = MiniMindConfig(
        hidden_size=args.teacher_size, 
        num_hidden_layers=16
    )
    lm_config_student = MiniMindConfig(
        hidden_size=args.student_size, 
        num_hidden_layers=8
    )
    
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
    
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=f"Distill-T{args.teacher_size}-S{args.student_size}")
    else:
        wandb = None

    # 初始化模型
    model, tokenizer = init_model(lm_config_student)
    teacher_model = init_model(lm_config_teacher, is_teacher=True)
    
    # 数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_ds) if ddp else None,
        pin_memory=True,
        num_workers=args.num_workers
    )

    # DeepSpeed初始化
    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
        model.parameters(), 
        lr=args.learning_rate,
        betas=(0.9, 0.95)
    )
    
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config="ds_config.json",
        model_parameters=model.parameters(),
        dist_init_required=ddp
    )

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb, alpha=args.alpha, temperature=args.temperature)
