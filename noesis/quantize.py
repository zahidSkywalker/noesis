"""Noesis quantizer — weight-only int8, per-output-channel scales.

Shrinks the brain ~4x (fp32 -> int8). The quantized file is a portable
npz: weights + scales; a future runtime (CPU / WASM) reads it directly.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch


def quantize_tensor(w: torch.Tensor):
    """Per-row symmetric int8 quantization. Returns (q int8 [rows, cols], scale fp32 [rows])."""
    w = w.detach().float()
    absmax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    scale = absmax / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(1)


def main():
    ap = argparse.ArgumentParser(description="Quantize a Noesis checkpoint")
    ap.add_argument("--run", required=True, help="run dir containing latest.pt")
    ap.add_argument("--out", default=None, help="output .npz (default: <run>/model_int8.npz)")
    args = ap.parse_args()

    ckpt = torch.load(os.path.join(args.run, "latest.pt"), map_location="cpu", weights_only=False)
    sd = ckpt["model"]

    out_path = args.out or os.path.join(args.run, "model_int8.npz")
    arrays, meta = {}, {}
    total_orig = total_q = 0
    for name, tensor in sd.items():
        t = tensor.detach().cpu()
        if t.ndim == 2 and "embed" not in name:
            q, s = quantize_tensor(t)
            arrays[name + ".q"] = q.numpy()
            arrays[name + ".s"] = s.numpy().astype(np.float32)
            total_orig += t.numel() * 4
            total_q += q.nbytes + s.nbytes
        else:
            arrays[name] = t.numpy()
            total_orig += t.numel() * 4
            total_q += t.numel() * (2 if t.dtype in (torch.bfloat16, torch.float16) else 4)
        meta[name] = {"shape": list(t.shape)}
    np.savez_compressed(out_path, **arrays)
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f)
    print(f"[quant] matrices quantized -> {out_path}")
    print(f"[quant] original ~{total_orig / 1e6:.1f} MB | int8 payload ~{total_q / 1e6:.1f} MB "
          f"(file on disk: {os.path.getsize(out_path) / 1e6:.1f} MB compressed)")


if __name__ == "__main__":
    main()
