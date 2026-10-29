"""Train the real Noesis BPE tokenizer on the corpus sample.

Usage:
  python tools/train_tokenizer.py --sample /path/corpus/sample.txt \
      --vocab 32768 --out artifacts/tokenizer32k
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from noesis.tokenizer import BPETokenizer  # noqa: E402

SPECIALS = ["<|doc|>", "<|eos|>"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    with open(args.sample, encoding="utf-8", errors="replace") as f:
        text = f.read()
    t_read = time.time() - t0
    print(f"[tok] sample loaded: {len(text)/1e6:.1f}M chars in {t_read:.1f}s", flush=True)

    tok = BPETokenizer.train(text, args.vocab, specials=SPECIALS, verbose=True)
    train_s = time.time() - t0

    os.makedirs(args.out, exist_ok=True)
    vocab_path = os.path.join(args.out, "vocab.json")
    tok.save(vocab_path)

    # quick fertility probe on a held-out slice of the same sample (tail 2%)
    probe = text[-len(text) // 50:]
    ids = tok.encode(probe)
    fertility = len(probe) / max(1, len(ids))

    stats = {
        "vocab_size": tok.vocab_size,
        "n_merges": len(tok.merges),
        "specials": tok.specials,
        "sample_chars": len(text),
        "sample_mb": round(len(text.encode("utf-8")) / 1e6, 1),
        "train_seconds": round(train_s, 1),
        "probe_fertility_chars_per_token": round(fertility, 3),
        "format": 2,
    }
    with open(os.path.join(args.out, "train_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[tok] DONE: {tok.vocab_size} tokens ({len(tok.merges)} merges + {len(SPECIALS)} specials) "
          f"in {train_s/60:.1f} min | probe fertility {fertility:.2f} chars/token", flush=True)


if __name__ == "__main__":
    main()
