"""Noesis tokenizer gauntlet — the acceptance test for the real 32k brain-mouth.

Checks:
  1. Lossless byte round-trip on hostile strings (unicode, emoji, CJK, controls).
  2. Special tokens: reserved ids, round-trip, encode splitting.
  3. Fertility (chars/token) on held-out corpus slices — wiki + books.
  4. Encode throughput (MB/s), cold and warm.
  5. Vocab sanity: size, no overlap between byte ids / merges / specials.

Usage:
  python tools/eval_tokenizer.py --vocab artifacts/tokenizer32k/vocab.json \
      --corpus /path/corpus --report artifacts/tokenizer32k/report.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from noesis.tokenizer import BPETokenizer  # noqa: E402

HOSTILE = [
    "Hello, 世界! emoji 🙂🔥 and symbols ×÷~` kept.",
    "tabs\tand\nnewlines\r\nwindows lines   multiple    spaces",
    "mixed 数字123 and números and ᚠᚢᚦ runes and Ω≈ç√∫",
    "a", "", " ", "\n", "\x00 boundary \x1f control",
    "very" * 500 + " long word without spaces",
    "😀😀😀 emoji runs 🎉🎊🐲",
    "Ω≈ç√∫˜µ≤≥÷ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 math ∑∫ π ℝ",
    "JSON {\"key\": [1,2,3], \"nested\": {\"a\": true}} and code: def f(x): return x**2",
]

NASTY_ROUNDTRIP = [
    "Special inline: A<|doc|>B<|eos|>C must survive",
    "<|doc|><|eos|><|doc|>",
    "text ending with special <|eos|>",
]


def read_tail_gz(path: str, nbytes: int = 2_000_000) -> str:
    """Approximate tail read of a gz text shard (decompress all, keep tail)."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        data = f.read()
    return data[-nbytes:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    tok = BPETokenizer.load(args.vocab)
    results = {"vocab_size": tok.vocab_size, "n_merges": len(tok.merges),
               "specials": tok.specials, "checks": {}}

    # ---- 1. hostile round-trips ------------------------------------------
    fails = []
    for s in HOSTILE:
        if tok.decode(tok.encode(s)) != s:
            fails.append(s[:40])
    results["checks"]["hostile_roundtrip"] = {"pass": not fails, "fails": fails}
    print(f"[1] hostile round-trip: {'PASS' if not fails else 'FAIL ' + str(fails)}")

    # ---- 2. specials ------------------------------------------------------
    sp_ok = True
    for s in NASTY_ROUNDTRIP:
        if tok.decode(tok.encode(s)) != s:
            sp_ok = False
    ids = tok.encode("one<|doc|>two<|eos|>")
    sp_ok = sp_ok and tok.specials["<|doc|>"] in ids and tok.specials["<|eos|>"] in ids
    sp_ok = sp_ok and len(ids) == 4  # 'one' + doc + 'two' + eos exactly
    results["checks"]["specials"] = {"pass": sp_ok}
    print(f"[2] specials reserve + split + round-trip: {'PASS' if sp_ok else 'FAIL'}")

    # ---- 3. fertility on held-out corpus ----------------------------------
    fert = {}
    for kind in ("wiki", "books"):
        shards = sorted(p for p in os.listdir(args.corpus) if p.startswith(kind) and p.endswith(".gz"))
        if not shards:
            continue
        text = read_tail_gz(os.path.join(args.corpus, shards[0]))
        ids = tok.encode(text)
        fert[kind] = round(len(text) / max(1, len(ids)), 3)
        print(f"[3] fertility {kind}: {fert[kind]:.2f} chars/token ({len(text)/1e6:.1f}M chars)")
    results["checks"]["fertility"] = {"pass": all(v >= 3.6 for v in fert.values()) and len(fert) == 2,
                                      "values": fert}

    # ---- 4. throughput ----------------------------------------------------
    shards = sorted(p for p in os.listdir(args.corpus) if p.startswith("wiki") and p.endswith(".gz"))
    text = read_tail_gz(os.path.join(args.corpus, shards[0]), 1_000_000)
    mb = len(text.encode("utf-8")) / 1e6
    t0 = time.time(); tok.encode(text); cold = mb / (time.time() - t0)
    t0 = time.time(); tok.encode(text); warm = mb / (time.time() - t0)
    results["checks"]["throughput_mbs"] = {"pass": warm >= 1.0, "cold": round(cold, 2), "warm": round(warm, 2)}
    print(f"[4] encode throughput: cold {cold:.2f} MB/s | warm {warm:.2f} MB/s")

    # ---- 5. vocab sanity --------------------------------------------------
    top_merge = 256 + len(tok.merges)
    ids_special = sorted(tok.specials.values())
    sanity = (tok.vocab_size == 32768
              and ids_special == list(range(top_merge, tok.vocab_size))
              and len(set(tok.merges)) == len(tok.merges))
    results["checks"]["vocab_sanity"] = {"pass": sanity}
    print(f"[5] vocab sanity (size, specials at top, unique merges): {'PASS' if sanity else 'FAIL'}")

    all_pass = all(v.get("pass") for v in results["checks"].values())
    results["verdict"] = "PASS 5/5" if all_pass else "FAIL"
    print(f"\n=== GAUNTLET VERDICT: {results['verdict']} ===")
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        json.dump(results, open(args.report, "w"), indent=2)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
