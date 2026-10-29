"""Noesis data pipeline — text corpus -> token stream -> training batches.

Stages:
  1. binarize:  text file(s) -> tokens via our BPE -> uint16 .bin (np.memmap)
  2. batches:   random context windows for training (deterministic via seed)
"""

from __future__ import annotations

import numpy as np
import torch


def binarize(tokenizer, text_paths: list[str], out_bin: str) -> int:
    """Encode text files into a single uint16 token stream. Returns n_tokens."""
    ids: list[int] = []
    step = 1 << 24  # 16M ids per flush to keep memory bounded
    n = 0
    with open(out_bin, "wb") as out:
        buf = np.empty(step, dtype=np.uint16)

        def flush(chunk: list[int]):
            nonlocal n
            while chunk:
                take = chunk[:step]
                buf[: len(take)] = take
                out.write(buf[: len(take)].tobytes())
                n += len(take)
                chunk = chunk[step:]

        for path in text_paths:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            flush(tokenizer.encode(text))
    return n


class TokenStream:
    """Random-window sampler over a uint16 token file."""

    def __init__(self, bin_path: str, seq_len: int, batch_size: int, seed: int = 1337):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        if len(self.data) < seq_len + 1:
            raise ValueError(f"corpus too small: {len(self.data)} tokens < {seq_len + 1}")

    def __len__(self):
        return len(self.data)

    def next_batch(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (inputs [B, T], targets [B, T]) — targets shifted by 1."""
        hi = len(self.data) - self.seq_len - 1
        starts = self.rng.integers(0, hi, size=self.batch_size)
        x = np.stack([self.data[s: s + self.seq_len] for s in starts]).astype(np.int64)
        y = np.stack([self.data[s + 1: s + self.seq_len + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
