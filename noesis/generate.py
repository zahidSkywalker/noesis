"""Noesis generation — streaming inference with KV-cache.

Sampling: temperature, top-p (nucleus), repetition penalty, seed.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from .model import ModelConfig, NoesisModel
from .tokenizer import BPETokenizer


@torch.no_grad()
def generate(model: NoesisModel, tokenizer: BPETokenizer, prompt: str, max_new_tokens: int,
             temperature: float = 0.8, top_p: float = 0.95, rep_penalty: float = 1.1,
             seed: int | None = None, device: str = "cpu", quiet: bool = False) -> str:
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()

    ids = tokenizer.encode(prompt) or tokenizer.encode(" ")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    caches = model.init_caches(1, device)

    out_ids = list(ids)
    logits, _ = model(x, caches=caches)          # full prompt (prefill)
    logits = logits[:, -1, :]

    for i in range(max_new_tokens):
        if rep_penalty != 1.0:
            for tid in set(out_ids[-64:]):
                logits[0, tid] /= rep_penalty
        if temperature <= 1e-6:                   # greedy
            nxt = int(torch.argmax(logits, dim=-1))
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                sp, si = torch.sort(probs[0], descending=True)
                cum = torch.cumsum(sp, dim=-1)
                cut = int((cum > top_p).nonzero()[0]) + 1 if bool((cum > top_p).any()) else len(sp)
                mask = torch.zeros_like(probs[0], dtype=torch.bool)
                mask[si[:cut]] = True
                probs = torch.where(mask[None, :], probs, torch.zeros_like(probs))
                probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs, 1))
        out_ids.append(nxt)
        if not quiet:
            print(tokenizer.decode([nxt]), end="", flush=True)
        if nxt >= tokenizer.vocab_size:
            break
        logits, _ = model(torch.tensor([[nxt]], device=device), caches=caches)
        logits = logits[:, -1, :]

    if not quiet:
        print()
    return tokenizer.decode(out_ids)


def load_model(run_dir: str, device: str = "cpu"):
    ckpt = torch.load(os.path.join(run_dir, "latest.pt"), map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["cfg"])
    model = NoesisModel(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    tok = BPETokenizer.load(ckpt["tokenizer_path"])
    return model, tok, ckpt.get("step", 0)


def main():
    ap = argparse.ArgumentParser(description="Noesis generation")
    ap.add_argument("--run", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--rep_penalty", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    model, tok, step = load_model(args.run, args.device)
    print(f"[gen] loaded run '{args.run}' (step {step}, {sum(p.numel() for p in model.parameters()):,} params)")
    print(f"[gen] prompt: {args.prompt!r}")
    print("---")
    generate(model, tok, args.prompt, args.tokens,
             temperature=0.0 if args.greedy else args.temperature,
             top_p=args.top_p, rep_penalty=args.rep_penalty,
             seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
