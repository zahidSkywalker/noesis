"""Noesis byte-level BPE tokenizer — 100% from scratch.

Design:
  - Byte-level: any UTF-8 text round-trips losslessly (no <unk> ever).
  - Base vocabulary: the 256 possible bytes (ids 0..255).
  - Merges learned from corpus with an incremental pair-count index.
  - Merge selection uses count BUCKETS (count -> set of pairs) with lazy
    verification, so finding the most frequent pair is amortized ~O(1)
    instead of a full scan. This is what makes 32k vocab training fast
    and memory-safe (each pair lives in exactly one bucket).
  - Pre-tokenization: words = optional leading space + non-space run,
    GPT-2-style but our own simplified rule. Merge learning never
    crosses word boundaries (keeps merges linguistically sane).
  - Special tokens (v2 format): reserved ids at the TOP of the vocab,
    e.g. <|doc|> (document separator) and <|eos|>. Encoding splits on
    them when they appear in text.

File format (vocab.json):
  v1: {"version": 1, "vocab_size": int, "merges": [[b64a, b64b], ...]}
  v2: same + "specials": {"<|doc|>": id, ...}   # ids at top of vocab
"""

from __future__ import annotations

import base64
import collections
import json
import re

_WORD_RE = re.compile(r" ?[^\s]+|\s+")


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


class BPETokenizer:
    def __init__(self, merges=None, specials: dict[str, int] | None = None):
        # merges: list of (left: bytes, right: bytes) in rank order
        self.merges: list[tuple[bytes, bytes]] = list(merges or [])
        self._ranks: dict[tuple[bytes, bytes], int] = {m: i for i, m in enumerate(self.merges)}
        self._word_cache: dict[bytes, list[int]] = {}
        self.specials: dict[str, int] = dict(specials or {})
        self._id_to_special: dict[int, str] = {v: k for k, v in self.specials.items()}
        self._bmap_cache: dict[bytes, int] | None = None

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.specials)

    def id_to_bytes(self, token_id: int) -> bytes:
        if token_id < 256:
            return bytes([token_id])
        if token_id >= 256 + len(self.merges):
            raise ValueError(f"token id {token_id} is a special or out of range")
        pair = self.merges[token_id - 256]
        return pair[0] + pair[1]

    def _bmap(self) -> dict[bytes, int]:
        if self._bmap_cache is None:
            self._bmap_cache = {a + c: 256 + i for i, (a, c) in enumerate(self.merges)}
        return self._bmap_cache

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int, specials: list[str] | None = None,
              verbose: bool = False) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        specials = list(specials or [])
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError("vocab_size too small for specials")

        # ---- unique-word byte sequences + corpus counts ----------------
        word_counts: collections.Counter[bytes] = collections.Counter()
        for m in _WORD_RE.finditer(text):
            word_counts[m.group(0).encode("utf-8")] += 1
        del text

        seqs: list[list] = []  # [symbols(list[bytes]), cnt(int)]
        for word, cnt in word_counts.items():
            seqs.append([[word[i:i + 1] for i in range(len(word))], cnt])
        del word_counts

        # ---- pair index -------------------------------------------------
        # counts[pair]  -> weighted frequency (summed over word occurrences)
        # buckets[cnt]  -> set of pairs with that count (max lookup in ~O(1))
        # index[pair]   -> set of seq ids whose CURRENT symbols contain the
        #                  pair adjacently; lets each merge touch only the
        #                  affected words instead of scanning all of them.
        counts: dict[tuple[bytes, bytes], int] = {}
        buckets: dict[int, set] = collections.defaultdict(set)
        index: dict[tuple[bytes, bytes], set] = collections.defaultdict(set)

        def bump(pair, delta: int) -> None:
            if not delta:
                return
            old = counts.get(pair, 0)
            new = old + delta
            if new > 0:
                if old > 0:
                    b = buckets[old]
                    b.discard(pair)
                    if not b:
                        del buckets[old]
                counts[pair] = new
                buckets[new].add(pair)
            else:
                if old > 0:
                    b = buckets[old]
                    b.discard(pair)
                    if not b:
                        del buckets[old]
                counts.pop(pair, None)

        for sid, (seq, cnt) in enumerate(seqs):
            for pair in zip(seq, seq[1:]):
                bump(pair, cnt)
                index[pair].add(sid)

        def pop_max():
            while buckets:
                top = max(buckets)  # O(#distinct counts) — small in practice
                s = buckets[top]
                while s:
                    pair = next(iter(s))
                    s.discard(pair)
                    real = counts.get(pair, 0)
                    if real == top:
                        counts.pop(pair, None)
                        return pair, real
                    if real > 0:
                        counts[pair] = real
                        buckets[real].add(pair)
                del buckets[top]
            return None, 0

        tok = cls()
        for step in range(n_merges):
            best, best_n = pop_max()
            if best is None or best_n < 2:
                if verbose:
                    print(f"[bpe] corpus exhausted at merge {step}")
                break
            a, b = best
            merged = a + b

            tok.merges.append((a, b))
            tok._ranks[(a, b)] = step
            tok._bmap_cache = None

            affected = index.pop(best, set())
            for sid in affected:
                seq, cnt = seqs[sid]
                old_occ: collections.Counter = collections.Counter(zip(seq, seq[1:]))
                i, out = 0, []
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                new_occ: collections.Counter = collections.Counter(zip(out, out[1:]))
                for pair, o in old_occ.items():
                    bump(pair, cnt * (new_occ.get(pair, 0) - o))
                    if pair not in new_occ:
                        s = index.get(pair)
                        if s is not None:
                            s.discard(sid)
                            if not s:
                                del index[pair]
                for pair, n in new_occ.items():
                    if pair not in old_occ:
                        bump(pair, cnt * n)  # newly created pairs earn their count
                        index[pair].add(sid)
                seqs[sid][0] = out

            if verbose and (step + 1) % max(1, n_merges // 10) == 0:
                print(f"[bpe] merge {step + 1}/{n_merges}: {a!r} + {b!r} -> {merged!r} "
                      f"(count {best_n}, touched {len(affected)} words)", flush=True)

        # ---- reserve special ids at the TOP of the vocab --------------
        base = vocab_size - len(specials)
        tok.specials = {s: base + i for i, s in enumerate(specials)}
        tok._id_to_special = {v: k for k, v in tok.specials.items()}
        tok._word_cache.clear()
        return tok

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------
    def _encode_word(self, word_bytes: bytes) -> list[int]:
        cached = self._word_cache.get(word_bytes)
        if cached is not None:
            return cached
        parts: list[bytes] = [word_bytes[i:i + 1] for i in range(len(word_bytes))]
        ranks = self._ranks
        while len(parts) > 1:
            best_rank, best_i = None, None
            for i in range(len(parts) - 1):
                r = ranks.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            merged = parts[best_i] + parts[best_i + 1]
            parts[best_i:best_i + 2] = [merged]
        bmap = self._bmap()
        ids = []
        for p in parts:
            if len(p) == 1:
                ids.append(p[0])
            else:
                ids.append(bmap[p])
        self._word_cache[word_bytes] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self.specials:
            # split on special tokens so they map to their reserved ids
            pattern = "(" + "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True)) + ")"
            for chunk in re.split(pattern, text):
                if not chunk:
                    continue
                sid = self.specials.get(chunk)
                if sid is not None:
                    ids.append(sid)
                else:
                    for m in _WORD_RE.finditer(chunk):
                        ids.extend(self._encode_word(m.group(0).encode("utf-8")))
        else:
            for m in _WORD_RE.finditer(text):
                ids.extend(self._encode_word(m.group(0).encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[bytes] = []
        for i in ids:
            i = int(i)
            sp = self._id_to_special.get(i)
            out.append(sp.encode("utf-8") if sp is not None else self.id_to_bytes(i))
        return b"".join(out).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "version": 2,
            "vocab_size": self.vocab_size,
            "merges": [[_b64(a), _b64(b)] for a, b in self.merges],
            "specials": self.specials,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        specials = payload.get("specials", {})
        tok = cls([(_unb64(a), _unb64(b)) for a, b in payload["merges"]], specials)
        return tok
