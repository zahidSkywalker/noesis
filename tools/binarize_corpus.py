"""Streamed corpus binarizer: .txt.gz shards + trained BPE -> uint16 tokens.bin

M4 tool — encodes with the EXISTING 32k tokenizer (prepare.py trains a new one,
which is NOT what we want for pretraining). One shard in memory at a time.

Usage (Kaggle or local):
  python tools/binarize_corpus.py \
      --shards "corpus/wiki-*.txt.gz" "corpus/books-*.txt.gz" \
      --tokenizer artifacts/tokenizer32k/vocab.json \
      --out /kaggle/working/tokens.bin --stats /kaggle/working/bin_stats.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import time

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noesis.tokenizer import BPETokenizer  # noqa: E402


def read_shard_text(p: str) -> str:
    """Tolerant shard read: .gz (with truncation salvage) or plain text.
    Kaggle auto-decompresses .gz dataset uploads, so both forms occur."""
    if not p.endswith(".gz"):
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    data = bytearray()
    truncated = False
    try:
        with gzip.open(p, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                data += chunk
    except EOFError:
        truncated = True
    if truncated:
        print(f"[bin] WARNING: {os.path.basename(p)} truncated — salvaging "
              f"{len(data):,} bytes prefix", flush=True)
    return data.decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser(description="Binarize gz corpus shards with a trained tokenizer")
    ap.add_argument("--shards", nargs="+", required=True, help="globs of .txt.gz shards")
    ap.add_argument("--tokenizer", required=True, help="vocab.json from tokenizer32k")
    ap.add_argument("--out", required=True, help="output tokens.bin (uint16)")
    ap.add_argument("--stats", default="", help="optional stats json path")
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.shards:
        hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"[bin] WARNING: no files match {pattern}")
        paths.extend(hits)
    if not paths:
        raise SystemExit("no shard files found")

    print(f"[bin] {len(paths)} shard(s); loading tokenizer {args.tokenizer}")
    tok = BPETokenizer.load(args.tokenizer)
    print(f"[bin] tokenizer ready: vocab {tok.vocab_size:,}")

    step = 1 << 24  # 16M ids per flush, bounded memory
    buf = np.empty(step, dtype=np.uint16)
    n_tokens = 0
    n_chars = 0
    t0 = time.time()

    with open(args.out, "wb") as out:
        for i, p in enumerate(paths, 1):
            text = read_shard_text(p)
            n_chars += len(text)
            ids = tok.encode(text)
            pos = 0
            while pos < len(ids):
                take = min(step, len(ids) - pos)
                buf[:take] = ids[pos: pos + take]
                out.write(buf[:take].tobytes())
                n_tokens += take
                pos += take
            dt = time.time() - t0
            print(f"[bin] {i}/{len(paths)} {os.path.basename(p)}: "
                  f"+{len(ids):,} tok | total {n_tokens:,} | {dt:,.0f}s "
                  f"| {n_chars / max(1, dt) / 1e6:.1f} MB/s", flush=True)

    stats = {
        "shards": len(paths),
        "chars": n_chars,
        "tokens": n_tokens,
        "chars_per_token": round(n_chars / max(1, n_tokens), 3),
        "approx_words": int(n_tokens * 0.75),
        "out_bytes": os.path.getsize(args.out),
        "seconds": round(time.time() - t0, 1),
    }
    print(f"[bin] DONE: {n_tokens:,} tokens | {stats['chars_per_token']} chars/tok "
          f"| {stats['out_bytes']:,} bytes")
    if args.stats:
        with open(args.stats, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
