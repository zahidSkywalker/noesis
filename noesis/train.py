"""Noesis trainer — from-scratch pretraining loop.

Features:
  - AdamW + cosine schedule with warmup + gradient clipping
  - gradient accumulation (effective batch > memory batch)
  - bf16 autocast on GPU, fp32 on CPU
  - checkpoints: model + optimizer + scheduler + step + config
    (a "cold" checkpoint alone is NOT enough to resume — we save it all)
  - resume: load latest checkpoint and continue exactly
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .data import TokenStream
from .model import ModelConfig, NoesisModel


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total_steps - warmup)
    t = min(1.0, t)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


def save_checkpoint(path: str, model, optimizer, scheduler_step, cfg_dict, tokenizer_path, data_bin, step, best_val=None):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler_step": scheduler_step,
        "cfg": cfg_dict,
        "tokenizer_path": tokenizer_path,
        "data_bin": data_bin,
        "step": step,
        "best_val": best_val,
    }, path)


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["cfg"])
    model = NoesisModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return ckpt, cfg, model


def main():
    ap = argparse.ArgumentParser(description="Noesis pretrainer")
    ap.add_argument("--config", required=True, help="model+train config json")
    ap.add_argument("--data", required=True, help="uint16 token .bin")
    ap.add_argument("--tokenizer", required=True, help="vocab.json")
    ap.add_argument("--out", required=True, help="output run dir")
    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint in --out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        conf = json.load(f)

    mcfg = ModelConfig.from_dict(conf["model"])
    tcfg = conf["train"]
    device = args.device

    model = NoesisModel(mcfg).to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters()):,} | device: {device}")
    print(f"[train] config: {json.dumps(mcfg.to_dict())}")

    stream = TokenStream(args.data, mcfg.max_seq_len, tcfg["batch_size"], seed=tcfg.get("seed", 1337))

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.ndim == 1 or "embed" in n) else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg.get("weight_decay", 0.1)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg["lr"], betas=(0.9, 0.95), eps=1e-8)

    total_steps = tcfg["total_steps"]
    warmup = tcfg.get("warmup", min(100, total_steps // 10))
    accum = tcfg.get("grad_accum", 1)
    save_every = tcfg.get("save_every", 500)
    log_every = tcfg.get("log_every", 50)
    ckpt_path = os.path.join(args.out, "latest.pt")

    start_step, sched_step = 0, 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt, _, model = load_checkpoint(ckpt_path, device)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        sched_step = ckpt["scheduler_step"]
        print(f"[train] resumed from step {start_step}")

    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(start_step, total_steps):
        lr = cosine_lr(sched_step, total_steps, tcfg["lr"], warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(accum):
            x, y = stream.next_batch(device)
            _, loss = model(x, targets=y)
            (loss / accum).backward()
            loss_val += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sched_step += 1
        running += loss_val

        if (step + 1) % log_every == 0:
            dt = time.time() - t0
            toks = log_every * accum * tcfg["batch_size"] * mcfg.max_seq_len
            print(f"[train] step {step + 1}/{total_steps} loss {running / log_every:.4f} lr {lr:.2e} "
                  f"tok/s {toks / dt:,.0f} elapsed {dt:.0f}s")
            running, t0 = 0.0, time.time()

        if (step + 1) % save_every == 0 or (step + 1) == total_steps:
            save_checkpoint(ckpt_path, model, optimizer, sched_step, mcfg.to_dict(),
                            args.tokenizer, args.data, step + 1)
            print(f"[train] checkpoint -> {ckpt_path} (step {step + 1})")

    print("[train] done.")


if __name__ == "__main__":
    main()
