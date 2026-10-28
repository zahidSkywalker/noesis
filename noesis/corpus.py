"""Noesis corpus builder — public data, self-hosted once, streamed through.

Sources (approved in the Noesis Bible):
  - wikimedia/wikipedia 20231101.en parquet shards (HuggingFace host; public data)
  - Project Gutenberg top-N most-downloaded English books

Design constraints (sandbox: ~6GB free disk, 3GB RAM, 2 cores):
  - STREAMING: download one unit -> extract -> clean -> dedup -> append to
    gzip text shard -> DELETE raw. The raw corpus never rests on disk.
  - Dedup: paragraph-level Bloom filter (own implementation, ~32MB fixed
    memory, catches repeated wiki boilerplate + repeated book text).
  - RESUME: state.json tracks finished units; safe to re-run anytime.
  - DISK GUARD: stops gracefully when free space drops below the floor.
  - SAMPLE: deterministic hash-based collector (every Nth doc) writes a
    capped plain-text sample for BPE tokenizer training.

Usage:
  python -m noesis.corpus --out /path/corpus --wiki-shards 20 --books 400 --sample-mb 160
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import os
import re
import time
import unicodedata

import requests

WIKI_URL = ("https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/"
            "20231101.en/train-{:05d}-of-00041.parquet")
GUT_CATALOG = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
GUT_BOOK = "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
UA = "NoesisCorpusBot/0.1 (from-scratch LLM research; zahidskywalker)"
DISK_FLOOR = 1_500_000_000  # stop building when free space < 1.5GB
SHARD_TEXT_BYTES = 64 << 20  # 64MB uncompressed text per .gz shard

_PG_MARK = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
_PG_END = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_URL_LINE = re.compile(r"^(https?://\S+|www\.\S+)$", re.I)


# ----------------------------------------------------------------------
# Bloom filter (own, fixed-memory paragraph dedup)
# ----------------------------------------------------------------------
class Bloom:
    def __init__(self, bits: int = 1 << 28, k: int = 7):  # 256Mbit = 32MB
        self.bits = bits
        self.k = k
        self.arr = bytearray(bits // 8)
        self.n = 0

    def _probes(self, item: str):
        h = hashlib.blake2b(item.encode("utf-8", "ignore"), digest_size=8).digest()
        h1 = int.from_bytes(h[:4], "little")
        h2 = int.from_bytes(h[4:], "little") | 1
        m = self.bits
        for i in range(self.k):
            yield (h1 + i * h2) % m

    def add(self, item: str) -> None:
        for p in self._probes(item):
            self.arr[p >> 3] |= 1 << (p & 7)
        self.n += 1

    def __contains__(self, item: str) -> bool:
        return all(self.arr[p >> 3] >> (p & 7) & 1 for p in self._probes(item))


# ----------------------------------------------------------------------
# cleaning
# ----------------------------------------------------------------------
def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = _CTRL_RE.sub("", t)
    lines = [ln.rstrip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln == "" or not _URL_LINE.match(ln)]
    t = "\n".join(lines)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def paragraphs(t: str, min_len: int = 40):
    for p in re.split(r"\n\s*\n", t):
        p = p.strip()
        if len(p) >= min_len:
            yield p


def doc_worth_keeping(t: str, min_doc: int) -> bool:
    if len(t) < min_doc:
        return False
    letters = sum(1 for c in t if c.isalnum() or c.isspace())
    return letters / max(1, len(t)) >= 0.70


def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=4).digest(), "little")


def save_json(path: str, obj) -> None:
    """Atomic write — a SIGTERM mid-dump can never corrupt the state file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ----------------------------------------------------------------------
# shard writer + sample collector
# ----------------------------------------------------------------------
class GzShards:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.idx = len(glob.glob(prefix + "-*.txt.gz"))
        self.path = None
        self.raw = None
        self.gz = None
        self.bytes_in_shard = 0

    def _open(self):
        self.path = f"{self.prefix}-{self.idx:04d}.txt.gz"
        self.raw = open(self.path, "wb")
        self.gz = gzip.GzipFile(fileobj=self.raw, mode="wb", compresslevel=6)
        self.bytes_in_shard = 0

    def write(self, text: str):
        if self.gz is None:
            self._open()
        data = text.encode("utf-8")
        self.gz.write(data)
        self.bytes_in_shard += len(data)
        if self.bytes_in_shard >= SHARD_TEXT_BYTES:
            self.rotate()

    def rotate(self):
        if self.gz:
            self.gz.close()
            self.raw.close()
            self.gz = self.raw = None
            self.idx += 1

    def paths(self):
        return sorted(glob.glob(self.prefix + "-*.txt.gz"))


class Sampler:
    """Deterministic hash-based doc sampling into a capped plain-text file."""

    def __init__(self, path: str, cap_bytes: int, rate_pct: int):
        self.path = path
        self.cap = cap_bytes
        self.rate = rate_pct
        self.written = 0
        self.f = open(path, "a", encoding="utf-8") if os.path.exists(path) else open(path, "w", encoding="utf-8")
        self.written = self.f.tell()

    def offer(self, text: str) -> bool:
        if self.written >= self.cap:
            return False
        if (stable_hash(text) % 100) >= self.rate:
            return False
        take = text[: self.cap - self.written]
        self.f.write(take + "\n\n")
        self.written += len(take) + 2
        return True

    def close(self):
        self.f.close()


# ----------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------
def fetch(url: str, dest: str, quiet: bool = False) -> bool:
    """Streamed, resumable download. Returns False on 404/missing."""
    tmp = dest + ".part"
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"User-Agent": UA}
    if pos:
        headers["Range"] = f"bytes={pos}-"
    r = requests.get(url, headers=headers, stream=True, timeout=90)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    mode = "ab" if pos and r.status_code == 206 else "wb"
    with open(tmp, mode) as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    if not quiet:
        print(f"    downloaded {os.path.getsize(dest)/1e6:.1f}MB -> {os.path.basename(dest)}", flush=True)
    return True


_FAMOUS_IDS = [1342, 84, 11, 2701, 1661, 98, 345, 2600, 1400, 46, 4300, 1080, 76, 174,
               219, 16, 55, 158, 43, 120, 1952, 205, 33, 41, 1232, 2542, 1250, 768,
               1497, 2554, 35, 36, 844, 863, 1727, 2814, 1184, 2148, 16328, 10, 3207,
               5200, 74, 271, 14988, 289, 39, 1257, 996, 1260, 45, 105, 100, 3296,
               2413, 135, 9296, 1922, 5827, 2000, 3600, 244, 5003, 155, 203, 730, 5147]
_KEEP_SHELVES = ("Best Books Ever Listings", "Harvard Classics",
                 "Bestsellers, American, 1895-1923", "Bestsellers, Historical Fiction")


def gutenberg_book_ids(n: int, cache_dir: str) -> list[int]:
    """Famous classics first, then a spread across curated Gutenberg bookshelves."""
    ids_file = os.path.join(cache_dir, "book_ids.json")
    if os.path.exists(ids_file):
        cached = json.load(open(ids_file))
        if cached:  # guard against a stale empty cache from a broken run
            return cached[:n]
        os.remove(ids_file)
    print("[books] fetching official catalog (one-time, ~21MB)...", flush=True)
    dest = os.path.join(cache_dir, "pg_catalog.csv")
    if not os.path.exists(dest):
        fetch(GUT_CATALOG, dest)
    curated = []
    with open(dest, encoding="utf-8-sig", newline="", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                if row.get("Type") != "Text" or not (row.get("Language") or "").lower().startswith("en"):
                    continue
                shelves = row.get("Bookshelves") or ""
                if any(k in shelves for k in _KEEP_SHELVES):
                    curated.append(int(row["Text#"]))
            except (ValueError, KeyError):
                continue
    os.remove(dest)
    curated = sorted(set(curated))
    step = max(1, len(curated) // max(1, n))
    spread = curated[::step]
    ids = list(dict.fromkeys(_FAMOUS_IDS + spread))  # famous first, deduped
    json.dump(ids, open(ids_file, "w"))
    print(f"[books] catalog parsed: {len(ids)} candidate ids", flush=True)
    return ids


def iter_gutenberg(path: str):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    m = _PG_MARK.search(raw)
    if not m:
        return
    start = m.end()
    m2 = _PG_END.search(raw, start)
    body = raw[start:m2.start()] if m2 else raw[start:]
    yield body


def iter_wiki(path: str):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=512, columns=["title", "text"]):
        titles = batch.column("title").to_pylist()
        texts = batch.column("text").to_pylist()
        for title, text in zip(titles, texts):
            yield f"{title}\n\n{text}" if title else text


# ----------------------------------------------------------------------
# main builder
# ----------------------------------------------------------------------
def build(out: str, wiki_shards: int, books: int, sample_mb: int, sample_rate: int = 12,
          max_seconds: int = 0):
    t_start = time.time()

    def time_up() -> bool:
        return max_seconds > 0 and (time.time() - t_start) > max_seconds

    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "_cache")
    os.makedirs(cache, exist_ok=True)
    state_path = os.path.join(out, "state.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {"books": [], "wiki": []}
    stats_path = os.path.join(out, "stats.json")
    if not os.path.exists(stats_path):
        save_json(stats_path, {"books": {"files": 0, "docs": 0, "chars": 0, "dups": 0},
                               "wiki": {"files": 0, "docs": 0, "chars": 0, "dups": 0}})
    stats = json.load(open(stats_path))

    bloom = Bloom()
    sampler = Sampler(os.path.join(out, "sample.txt"), sample_mb << 20, sample_rate)
    shard_of = {
        "wiki": GzShards(os.path.join(out, "wiki")),
        "books": GzShards(os.path.join(out, "books")),
    }
    wiki_sh = shard_of["wiki"]
    book_sh = shard_of["books"]

    def room() -> bool:
        free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
        return free > DISK_FLOOR

    def eat(text: str, src: str, min_doc: int) -> None:
        s = stats[src]
        sh = shard_of[src]
        s["docs"] += 1
        if not doc_worth_keeping(text, min_doc):
            return
        kept_any = False
        for p in paragraphs(clean_text(text)):
            if p in bloom:
                s["dups"] += 1
                continue
            bloom.add(p)
            sh.write(p + "\n\n")
            sampler.offer(p + "\n\n")
            kept_any = True
        if kept_any:
            s["chars"] += len(text)

    try:
        # ---------------- books first (fills sample with fiction) ----------
        ids = gutenberg_book_ids(books, cache)
        print(f"[books] target {books} books", flush=True)
        for gid in ids:
            if len(state["books"]) >= books or time_up():
                if time_up():
                    print(f"[chunk] time budget reached — clean exit "
                          f"({time.time()-t_start:.0f}s)", flush=True)
                break
            if gid in state["books"] or not room():
                if not room():
                    print("[disk] floor reached — stopping (resume later)", flush=True)
                    break
                continue
            dest = os.path.join(cache, f"pg{gid}.txt")
            try:
                if fetch(GUT_BOOK.format(gid=gid), dest, quiet=True):
                    for body in iter_gutenberg(dest):
                        eat(body, "books", min_doc=200)
                state["books"].append(gid)
                os.remove(dest)
                if len(state["books"]) % 10 == 0:
                    print(f"[books] {len(state['books'])}/{books} done | sample {sampler.written>>20}MB | "
                          f"chars {stats['books']['chars']/1e6:.0f}M | dups {stats['books']['dups']}", flush=True)
            except Exception as e:
                print(f"[books] id {gid} failed: {type(e).__name__}: {e}", flush=True)
                state["books"].append(gid)  # don't retry forever
            save_json(state_path, state)  # per-book: crash-proof resume
            time.sleep(0.4)  # polite to gutenberg.org

        # ---------------- wikipedia shards ---------------------------------
        n_done = 0
        for i in range(wiki_shards):
            if time_up():
                print(f"[chunk] time budget reached — clean exit "
                      f"({time.time()-t_start:.0f}s)", flush=True)
                break
            if not room():
                print("[disk] floor reached — stopping (resume later)", flush=True)
                break
            if i in state["wiki"]:
                continue
            print(f"[wiki] shard {i:05d} ({n_done}/{wiki_shards} this session)", flush=True)
            dest = os.path.join(cache, f"wiki-{i:05d}.parquet")
            try:
                if not fetch(WIKI_URL.format(i), dest):
                    state["wiki"].append(i)
                    continue
                for doc in iter_wiki(dest):
                    eat(doc, "wiki", min_doc=300)
                state["wiki"].append(i)
                os.remove(dest)
                n_done += 1
                print(f"    sample {sampler.written>>20}MB | kept chars {stats['wiki']['chars']/1e6:.0f}M | "
                      f"dup paras {stats['wiki']['dups']}", flush=True)
            except Exception as e:
                print(f"[wiki] shard {i} failed: {type(e).__name__}: {e}", flush=True)
                save_json(state_path, state)
                save_json(stats_path, stats)
            finally:
                if os.path.exists(dest + ".part"):
                    os.remove(dest + ".part")
                if os.path.exists(dest):
                    try:
                        os.remove(dest)
                    except OSError:
                        pass
            save_json(state_path, state)
            save_json(stats_path, stats)
    finally:
        wiki_sh.rotate()
        book_sh.rotate()
        sampler.close()
        save_json(state_path, state)
        save_json(stats_path, stats)

    total_chars = stats["books"]["chars"] + stats["wiki"]["chars"]
    manifest = {
        "wiki_shards_done": sorted(state["wiki"]),
        "books_done": len(state["books"]),
        "wiki_shards": wiki_sh.paths(),
        "books_shards": book_sh.paths(),
        "sample": os.path.join(out, "sample.txt"),
        "sample_bytes": sampler.written,
        "stats": stats,
        "total_chars": total_chars,
        "token_estimate": int(total_chars / 4.4),
    }
    save_json(os.path.join(out, "manifest.json"), manifest)
    print(f"[done] {total_chars/1e9:.2f}G chars kept | est {total_chars/4.4/1e9:.2f}B tokens | "
          f"sample {sampler.written>>20}MB | state saved", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Build the Noesis corpus")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiki-shards", type=int, default=20)
    ap.add_argument("--books", type=int, default=400)
    ap.add_argument("--sample-mb", type=int, default=160)
    ap.add_argument("--sample-rate", type=int, default=12)
    ap.add_argument("--max-seconds", type=int, default=0,
                    help="clean self-exit after N seconds (chunked execution)")
    args = ap.parse_args()
    build(args.out, args.wiki_shards, args.books, args.sample_mb, args.sample_rate,
          args.max_seconds)


if __name__ == "__main__":
    main()
