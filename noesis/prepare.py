"""Noesis corpus preparation — text files -> trained tokenizer + token bin.

Usage:
  python -m noesis.prepare --corpus corpus/book1.txt corpus/book2.txt \
                           --vocab 2048 --out runs/nano
Outputs:
  <out>/vocab.json   trained BPE tokenizer
  <out>/tokens.bin   uint16 token stream
  <out>/stats.json   corpus stats
"""

from __future__ import annotations

import argparse
import glob
import json
import os

from .data import binarize
from .tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser(description="Prepare Noesis corpus")
    ap.add_argument("--corpus", nargs="+", required=True, help="text files or globs")
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.corpus:
        hits = sorted(glob.glob(pattern))
        paths.extend(hits if hits else [pattern])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise SystemExit("no corpus files found")

    texts = []
    total_chars = 0
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            t = f.read()
        texts.append(t)
        total_chars += len(t)
    big_text = "\n".join(texts)

    os.makedirs(args.out, exist_ok=True)
    print(f"[prepare] corpus: {len(paths)} file(s), {total_chars:,} chars")
    tok = BPETokenizer.train(big_text, args.vocab, verbose=True)
    vocab_path = os.path.join(args.out, "vocab.json")
    tok.save(vocab_path)
    print(f"[prepare] tokenizer born: {tok.vocab_size} tokens -> {vocab_path}")

    bin_path = os.path.join(args.out, "tokens.bin")
    n = binarize(tok, paths, bin_path)
    approx_words = total_chars / 5.0
    stats = {
        "files": paths,
        "chars": total_chars,
        "tokens": n,
        "vocab_size": tok.vocab_size,
        "compression_chars_per_token": round(total_chars / max(1, n), 2),
        "approx_words": int(approx_words),
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[prepare] {n:,} tokens -> {bin_path}")
    print(f"[prepare] compression: {stats['compression_chars_per_token']} chars/token")


if __name__ == "__main__":
    main()
