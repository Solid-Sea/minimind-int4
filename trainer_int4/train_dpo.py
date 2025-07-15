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
from dataset.lm_dataset import DPODataset
from bitsandbytes.nn import Linear4bit
import deepspeed

warnings.filterwarnings('ignore')

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def logits_to_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs

def dpo_loss(ref_probs, probs, mask, beta):
    seq_lengths = mask.sum(dim=1, keepdim=True)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:]

    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    return loss

def train_epoch(epoch, wandb):
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            # 计算策略模型输出
            res = model(X)
            probs = logits_to_probs(res.logits, Y)
            
            # 计算参考模型输出
            with torch.no_grad():
                ref_res = ref_model(X)
                ref_probs = logits_to_probs(ref_res.logits, Y)
            
            # 计算DPO损失
            loss = dpo_loss(ref_probs, probs, loss_mask, beta=0.1)
            loss = loss / args.accumulation_steps

        # DeepSpeed梯度处理
        model.backward(loss)
        model.step()

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iter_per_epoch}) loss:{loss.item()*args.accumulation_steps:.3f} lr:{optimizer.param_groups[-1]["lr"]:.12f}')
            
            if wandb and (not ddp or dist.get_rank() == 0):
                wandb.log({
                    "loss": loss * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/rlhf_{lm_config.hidden_size}{moe_path}.pth'
            
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            torch.save({k: v.half() for k, v in state_dict.items()}, ckp)
            model.train()

def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    
    # 初始化策略模型
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
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{"_moe" if lm_config.use_moe else ""}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    
    # 初始化参考模型（冻结参数）
    ref_model = MiniMindForCausalLM(lm_config)
    ref_model.load_state_dict(state_dict)
    ref_model.eval()
    ref_model.requires_grad_(False)
    
    # 应用量化
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)
    
    Logger(f'可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M')
    return model, ref_model, tokenizer

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind RLHF")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RLHF-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl")

    args = parser.parse_args()
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=args.use_moe
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
        wandb.init(project=args.wandb_project, name=f"DPO-E{args.epochs}-BS{args.batch_size}")
    else:
        wandb = None

    model, ref_model, tokenizer = init_model(lm_config)
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
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
        train_epoch(epoch, wandb)
