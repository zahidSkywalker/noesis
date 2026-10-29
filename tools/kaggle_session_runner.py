"""Noesis M4 — Kaggle pretrain session runner.

Pattern: kernel-output chaining.
  Session 1: dataset input only -> binarize corpus -> train from step 0
  Session N: prior session's Output mounted at /kaggle/input/noesis-pretrain
             -> reuse tokens.bin + latest.pt -> train --resume

After the session, /kaggle/working (tokens.bin + run/latest.pt) becomes this
version's Output; the next push chains it as input again.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

IN = "/kaggle/input"
WORK = "/kaggle/working"
RUN = os.path.join(WORK, "run")
MAX_HOURS = 8.2  # Kaggle GPU sessions cap at 9h — exit clean before the wall

t0 = time.time()
os.makedirs(RUN, exist_ok=True)


def log(msg):
    print(f"[k {time.time() - t0:7.0f}s] {msg}", flush=True)


def find_one(pattern, required=True, recursive=False):
    hits = sorted(glob.glob(pattern, recursive=recursive))
    if hits:
        return hits[0]
    if required:
        raise SystemExit(f"[k] FATAL: nothing matches {pattern}")
    return None


# ---------------------------------------------------------------- inputs
log(f"input mounts: {os.listdir(IN)}")

import zipfile  # noqa: E402

ds = find_one(f"{IN}/noesis-m4*")
log(f"dataset dir: {ds} ({os.listdir(ds)})")

# dataset parts may arrive as subdirs (dir-mode skip) or .zip bundles (dir-mode zip)
TMP = "/kaggle/tmp"
os.makedirs(TMP, exist_ok=True)
for z in glob.glob(f"{ds}/*.zip"):
    name = os.path.basename(z)[:-4]
    dest = os.path.join(TMP, name)
    if not os.path.exists(os.path.join(dest, ".unpacked")):
        log(f"unzipping {name} -> {dest}")
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        open(os.path.join(dest, ".unpacked"), "w").close()
log(f"tmp contents: {os.listdir(TMP)}")

def part(name):
    """Locate a dataset part dir, zip-unpacked or plain."""
    for cand in (os.path.join(TMP, name), f"{ds}/{name}", f"{ds}/{name}.unpacked"):
        if os.path.isdir(cand):
            return cand
    return None

code_dir = part("noesis_code")
if code_dir is None:
    code_dir = find_one(f"{ds}/noesis_code*", required=False)
if code_dir is None:
    raise SystemExit("[k] FATAL: noesis_code part not found")
# handle nested extraction: zip contained a top-level dir of the same name
# (noesis_code.zip -> noesis_code/noesis_code/...), so descend until the
# 'noesis' python package is directly inside
for _ in range(3):
    if os.path.isdir(os.path.join(code_dir, "noesis")):
        break
    inner = os.path.join(code_dir, "noesis_code")
    if os.path.isdir(os.path.join(inner, "noesis")):
        code_dir = inner
        log(f"nested code dir detected -> {code_dir}")
    else:
        break
log(f"code dir: {code_dir}")
sys.path.insert(0, code_dir)
os.chdir(code_dir)

tok = find_one(f"{part('tokenizer32k') or TMP}/vocab.json", required=False) \
    or find_one(f"{ds}/**/vocab.json", recursive=True)
log(f"tokenizer: {tok}")

shards = sorted(glob.glob(f"{part('corpus') or TMP}/*.txt.gz")) \
    or sorted(glob.glob(f"{ds}/*.txt")) \
    or sorted(glob.glob(f"{ds}/*.txt.gz")) \
    or sorted(glob.glob(f"{ds}/**/*.txt*", recursive=True))
log(f"corpus shards: {len(shards)}")
if not shards:
    raise SystemExit("[k] FATAL: no corpus shards in dataset")

# Kaggle extracts each .gz upload into its own DIRECTORY — expand dirs to
# the actual files inside
real_shards = []
for p in shards:
    if os.path.isdir(p):
        inner = [x for x in sorted(glob.glob(os.path.join(p, "*")))
                 if os.path.isfile(x)]
        if not inner:
            log(f"WARNING: empty shard dir {p}")
        real_shards.extend(inner)
    else:
        real_shards.append(p)
shards = real_shards
log(f"shards after dir expansion: {len(shards)} (e.g. {os.path.basename(shards[0])})")
if not shards:
    raise SystemExit("[k] FATAL: no shard files after expansion")

prev = None
for d in sorted(glob.glob(f"{IN}/*")):
    if "pretrain" in os.path.basename(d).lower() and os.path.basename(d) != os.path.basename(ds):
        prev = d
log(f"previous session output: {prev}")

# ---------------------------------------------------------------- tokens.bin
tokens = os.path.join(WORK, "tokens.bin")
if prev and os.path.exists(os.path.join(prev, "tokens.bin")):
    log("reusing tokens.bin from previous session output")
    if not os.path.exists(tokens):
        shutil.copy2(os.path.join(prev, "tokens.bin"), tokens)
elif os.path.exists(tokens):
    log("tokens.bin already in working dir")
else:
    log(f"binarizing {len(shards)} shards -> tokens.bin (this takes ~30-40 min)")
    r = subprocess.run(
        [sys.executable, "tools/binarize_corpus.py",
         "--shards", *shards,
         "--tokenizer", tok,
         "--out", tokens,
         "--stats", os.path.join(WORK, "bin_stats.json")])
    if r.returncode != 0 or not os.path.exists(tokens):
        raise SystemExit(f"[k] FATAL: binarization failed rc={r.returncode}")

# ---------------------------------------------------------------- checkpoint
prev_ckpt = os.path.join(prev, "run", "latest.pt") if prev else None
if prev_ckpt and os.path.exists(prev_ckpt):
    shutil.copy2(prev_ckpt, os.path.join(RUN, "latest.pt"))
    log(f"resuming from previous checkpoint: {prev_ckpt}")
else:
    log("no previous checkpoint — fresh start")

# ---------------------------------------------------------------- train
cmd = [sys.executable, "-m", "noesis.train",
       "--config", "configs/base_125m_kaggle.json",
       "--data", tokens,
       "--tokenizer", tok,
       "--out", RUN,
       "--resume",
       "--max-hours", str(MAX_HOURS)]
log(f"train cmd: {' '.join(cmd)}")
r = subprocess.run(cmd)
log(f"train exit code: {r.returncode}")

# ---------------------------------------------------------------- report
lp = os.path.join(RUN, "latest.pt")
if os.path.exists(lp):
    log(f"checkpoint size: {os.path.getsize(lp):,} bytes")
log(f"working dir contents: {os.listdir(WORK)}")
log(f"session wall time: {(time.time() - t0) / 3600:.2f}h")
log("done — save this version so the next session can chain it")
