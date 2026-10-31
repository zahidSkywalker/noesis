"""Noesis trainer — from-scratch pretraining loop.

Features:
  - AdamW + cosine schedule with warmup + gradient clipping
  - gradient accumulation (effective batch > memory batch)
  - AMP: config train.amp = "fp16" | "bf16" on CUDA (fp16 + GradScaler for T4),
    absent => plain fp32 (CPU/QA path unchanged)
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

from .data import TokenStream, token_count
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
    """Standalone loader for inspection/eval tooling.

    WARNING: returns a NEW model object. Do NOT use this to resume training —
    train.main() must torch.load + model.load_state_dict into the live model
    instead, or the optimizer ends up bound to orphaned parameters.
    """
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
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="clean self-exit after N hours: checkpoint, then stop (0 = run to total_steps)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t_session = time.time()

    os.makedirs(args.out, exist_ok=True)
    with open(args.config, encoding="utf-8") as f:
        conf = json.load(f)

    mcfg = ModelConfig.from_dict(conf["model"])
    tcfg = conf["train"]
    device = args.device

    model = NoesisModel(mcfg).to(device)
    print(f"[train] params: {sum(p.numel() for p in model.parameters()):,} | device: {device}")
    print(f"[train] config: {json.dumps(mcfg.to_dict())}")

    # ---- val split (config-driven; val_frac absent/0.0 => exactly the old path)
    # Held-out tail of tokens.bin, applied at SAMPLING time: tokens.bin itself
    # stays byte-identical across sessions (kernel-output chaining relies on
    # that). Train stream windows can never cross val_start; the val stream
    # sees only the tail, from a fixed seed, so val_loss is comparable across
    # evals AND across sessions.
    seed = tcfg.get("seed", 1337)
    val_frac = float(tcfg.get("val_frac", 0.0) or 0.0)
    val_every = int(tcfg.get("val_every", 500))
    val_batches = int(tcfg.get("val_batches", 16))
    n_tokens = token_count(args.data)
    val_start = int(n_tokens * (1.0 - val_frac)) if val_frac > 0 else n_tokens
    use_val = val_frac > 0 and (n_tokens - val_start) >= mcfg.max_seq_len * 2
    if val_frac > 0 and not use_val:
        print(f"[train] WARNING: val region too small ({n_tokens - val_start:,} tokens "
              f"for seq_len {mcfg.max_seq_len}) — val disabled, train sees everything")
        val_start = n_tokens
    stream = TokenStream(args.data, mcfg.max_seq_len, tcfg["batch_size"], seed=seed,
                         end=val_start if use_val else None)
    val_stream = None
    if use_val:
        val_stream = TokenStream(args.data, mcfg.max_seq_len, tcfg["batch_size"],
                                 seed=seed + 1, start=val_start)
        print(f"[train] val split: holding out last {n_tokens - val_start:,}/{n_tokens:,} "
              f"tokens ({val_frac * 100:.2f}%) — eval every {val_every} steps, "
              f"{val_batches} batches x {tcfg['batch_size'] * mcfg.max_seq_len:,} tok")

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

    amp = tcfg.get("amp")
    use_amp = bool(amp) and device == "cuda"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    if amp and amp_dtype is None:
        raise SystemExit(f"[train] unknown train.amp: {amp!r} (use fp16|bf16)")
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and device == "cuda"))
    if amp:
        print(f"[train] AMP: {amp} on {device} (scaler {'on' if scaler.is_enabled() else 'off'})")

    start_step, sched_step = 0, 0
    best_val = None
    if args.resume and os.path.exists(ckpt_path):
        # Resume must load weights into the SAME model object the optimizer was
        # built on. Never rebind `model` here: a fresh object would take the
        # backward grads while the optimizer keeps stepping orphaned tensors
        # (GradScaler: "No inf checks were recorded for this optimizer").
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        sched_step = ckpt["scheduler_step"]
        best_val = ckpt.get("best_val")  # None on pre-val checkpoints — fine
        print(f"[train] resumed from step {start_step}")

    # Identity guard: every live model parameter must be the exact tensor the
    # optimizer holds. Cheap (a few hundred id lookups) and turns any future
    # model/optimizer split into a loud FATAL before GPU money is burned.
    opt_params = {id(p) for g in optimizer.param_groups for p in g["params"]}
    if not all(id(p) in opt_params for p in model.parameters()):
        raise SystemExit("[train] FATAL: optimizer does not reference the live model's parameters")

    model.train()

    def run_val() -> float:
        """Fixed-window val loss (no grad, no RNG disturbance to the train stream)."""
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for _ in range(val_batches):
                x, y = val_stream.next_batch(device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    _, vloss = model(x, targets=y)
                tot += vloss.item()
        model.train()
        return tot / val_batches

    if val_stream is not None:
        best_val = run_val()
        print(f"[val] step {start_step} val_loss {best_val:.4f} (baseline)")

    t0 = time.time()
    running = 0.0
    for step in range(start_step, total_steps):
        if args.max_hours > 0 and (time.time() - t_session) > args.max_hours * 3600:
            save_checkpoint(ckpt_path, model, optimizer, sched_step, mcfg.to_dict(),
                            args.tokenizer, args.data, step, best_val=best_val)
            print(f"[train] session time limit hit at step {step} — checkpoint saved, exiting clean")
            break

        lr = cosine_lr(sched_step, total_steps, tcfg["lr"], warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_val = 0.0
        for _ in range(accum):
            x, y = stream.next_batch(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, targets=y)
            scaler.scale(loss / accum).backward()
            loss_val += loss.item() / accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        sched_step += 1
        running += loss_val

        if (step + 1) % log_every == 0:
            dt = time.time() - t0
            toks = log_every * accum * tcfg["batch_size"] * mcfg.max_seq_len
            print(f"[train] step {step + 1}/{total_steps} loss {running / log_every:.4f} lr {lr:.2e} "
                  f"tok/s {toks / dt:,.0f} elapsed {dt:.0f}s")
            running, t0 = 0.0, time.time()

        if val_stream is not None and (step + 1) % val_every == 0:
            v = run_val()
            best_val = v if best_val is None else min(best_val, v)
            print(f"[val] step {step + 1} val_loss {v:.4f} (best {best_val:.4f})")

        if (step + 1) % save_every == 0 or (step + 1) == total_steps:
            save_checkpoint(ckpt_path, model, optimizer, sched_step, mcfg.to_dict(),
                            args.tokenizer, args.data, step + 1, best_val=best_val)
            print(f"[train] checkpoint -> {ckpt_path} (step {step + 1})")

    print("[train] done.")
    if args.max_hours > 0:
        print(f"[train] session wall time: {(time.time() - t_session) / 3600:.2f}h")


if __name__ == "__main__":
    main()
