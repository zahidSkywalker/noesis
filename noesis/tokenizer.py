"""Noesis byte-level BPE tokenizer — 100% from scratch.

Design:
  - Byte-level: any UTF-8 text round-trips losslessly (no <unk> ever).
  - Base vocabulary: the 256 possible bytes (ids 0..255).
  - Merges learned from corpus with an incremental pair-count index,
    so training stays fast even for large vocabularies.
  - Pre-tokenization: words = optional leading space + non-space run,
    GPT-2-style but our own simplified rule. Merge learning never
    crosses word boundaries (keeps merges linguistically sane).

File format (vocab.json):
  {
    "version": 1,
    "vocab_size": int,             # 256 + n_merges
    "merges": [[b64a, b64b], ...]  # ordered by rank; id = 256 + rank
  }
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
    def __init__(self, merges=None):
        # merges: list of (left: bytes, right: bytes) in rank order
        self.merges: list[tuple[bytes, bytes]] = list(merges or [])
        self._ranks: dict[tuple[bytes, bytes], int] = {m: i for i, m in enumerate(self.merges)}
        self._word_cache: dict[bytes, list[int]] = {}

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def id_to_bytes(self, token_id: int) -> bytes:
        if token_id < 256:
            return bytes([token_id])
        if token_id >= self.vocab_size:
            raise ValueError(f"token id {token_id} out of range")
        pair = self.merges[token_id - 256]
        return pair[0] + pair[1]

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")

        # ---- build unique-word byte sequences + counts ----------------
        word_counts: collections.Counter[bytes] = collections.Counter()
        for m in _WORD_RE.finditer(text):
            word_counts[m.group(0).encode("utf-8")] += 1

        # symbol sequences as list[list[bytes]] with occurrence counts
        seqs: list[tuple[list[bytes], int]] = []
        for word, cnt in word_counts.items():
            seqs.append(([word[i:i + 1] for i in range(len(word))], cnt))

        # ---- incremental pair counts ----------------------------------
        pair_counts: collections.Counter[tuple[bytes, bytes]] = collections.Counter()
        for seq, cnt in seqs:
            for a, b in zip(seq, seq[1:]):
                pair_counts[(a, b)] += cnt

        tok = cls()
        n_merges = vocab_size - 256
        for step in range(n_merges):
            if not pair_counts:
                if verbose:
                    print(f"[bpe] corpus exhausted at merge {step}")
                break
            best, best_n = pair_counts.most_common(1)[0]
            if best_n < 2:
                if verbose:
                    print(f"[bpe] best pair count {best_n} < 2 at merge {step}; stopping")
                break
            (a, b), merged = best, best[0] + best[1]

            # learn merge, drop stale pair
            tok.merges.append((a, b))
            tok._ranks[(a, b)] = step
            del pair_counts[best]

            # rewrite only sequences that contain the pair
            new_seqs = []
            for seq, cnt in seqs:
                if a not in seq or b not in seq:
                    new_seqs.append((seq, cnt))
                    continue
                i, out = 0, []
                hit = False
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                        if i > 0:
                            left = (seq[i - 1], a)
                            pair_counts[left] -= cnt
                            if pair_counts[left] <= 0:
                                del pair_counts[left]
                            pair_counts[(seq[i - 1], merged)] += cnt
                        if i < len(seq) - 2:
                            right = (b, seq[i + 2])
                            pair_counts[right] -= cnt
                            if pair_counts[right] <= 0:
                                del pair_counts[right]
                            pair_counts[(merged, seq[i + 2])] += cnt
                        out.append(merged)
                        i += 2
                        hit = True
                    else:
                        out.append(seq[i])
                        i += 1
                new_seqs.append((out, cnt))
            seqs = new_seqs
            if verbose and (step + 1) % max(1, n_merges // 10) == 0:
                print(f"[bpe] merge {step + 1}/{n_merges}: {a!r} + {b!r} -> {merged!r} (count {best_n})")

        tok._word_cache.clear()
        return tok

    # ------------------------------------------------------------------
    # encode / decode
    # ------------------------------------------------------------------
    def _bmap(self):
        return {a + c: 256 + i for i, (a, c) in enumerate(self.merges)}

    def _encode_word(self, word_bytes: bytes) -> list[int]:
        cached = self._word_cache.get(word_bytes)
        if cached is not None:
            return cached
        parts: list[bytes] = [word_bytes[i:i + 1] for i in range(len(word_bytes))]
        while len(parts) > 1:
            best_rank, best_i = None, None
            for i in range(len(parts) - 1):
                r = self._ranks.get((parts[i], parts[i + 1]))
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
        for m in _WORD_RE.finditer(text):
            ids.extend(self._encode_word(m.group(0).encode("utf-8")))
        return ids

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.id_to_bytes(int(i)) for i in ids)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        payload = {
            "version": 1,
            "vocab_size": self.vocab_size,
            "merges": [[_b64(a), _b64(b)] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls([(_unb64(a), _unb64(b)) for a, b in payload["merges"]])
        return tok
