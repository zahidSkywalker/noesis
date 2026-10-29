"""Noesis M4 — Kaggle pretrain session runner.

Pattern: kernel-output chaining.
  Session 1: dataset input only -> binarize corpus -> train from step 0
  Session N: prior session's Output mounted at /kaggle/input/noesis-pretrain
             -> reuse tokens.bin + latest.pt -> train --resume

After the session, /kaggle/working (tokens.bin + run/latest.pt) becomes this
version's Output; the next push chains it as input again.
"""
import glob
import hashlib
import json
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
# scan EVERY input mount, not just the corpus dataset — lets a small
# "noesis-code" dataset hot-swap the code without re-uploading 2.6GB of corpus
TMP = "/kaggle/tmp"
os.makedirs(TMP, exist_ok=True)
for mount in sorted(glob.glob(f"{IN}/*")):
    for z in sorted(glob.glob(f"{mount}/*.zip")):
        name = os.path.basename(z)[:-4]
        dest = os.path.join(TMP, name)
        if not os.path.exists(os.path.join(dest, ".unpacked")):
            log(f"unzipping {os.path.basename(mount)}/{name} -> {dest}")
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


def find_hot_code():
    """Locate the FIXED code in the noesis-code dataset mount.

    Kaggle extracts uploaded .zip archives at STORE time, so the code arrives
    as plain files (v6 lesson: TMP stayed empty and the runner fell back to
    the stale noesis-m4 copy). Walk the code mount a few levels deep and
    return the dir that directly contains the noesis package."""
    mounts = sorted(glob.glob(f"{IN}/*"))
    mounts.sort(key=lambda p: 0 if (
        "code" in os.path.basename(p).lower()
        and "noesis-m4" not in os.path.basename(p).lower()) else 1)
    for mount in mounts:
        base = os.path.basename(mount).lower()
        if "code" not in base or "noesis-m4" in base:
            continue
        for dirpath, dirnames, _files in os.walk(mount):
            if dirpath[len(mount):].count(os.sep) > 3:
                dirnames[:] = []
                continue
            if "noesis" in dirnames and os.path.isfile(os.path.join(dirpath, "noesis", "model.py")):
                return dirpath
    return None


code_dir = find_hot_code()
if code_dir is not None:
    log(f"code: hot-swap noesis-code dataset -> {code_dir}")
if code_dir is None:
    hot = os.path.join(TMP, "code", "noesis_code")
    code_dir = hot if os.path.isdir(os.path.join(hot, "noesis")) else None
    if code_dir is not None:
        log("code: using hot-swap noesis-code dataset (kept-as-zip)")
if code_dir is None:
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

# stamp: make stale-code runs IMPOSSIBLE to miss in the log
try:
    with open(os.path.join(code_dir, "noesis", "model.py"), "rb") as fh:
        _md5 = hashlib.md5(fh.read()).hexdigest()[:10]
    with open(os.path.join(code_dir, "configs", "base_125m_kaggle.json")) as fh:
        _nkv = json.load(fh)["model"]["n_kv_heads"]
    log(f"code stamp: model.py md5={_md5} n_kv_heads={_nkv}")
    if _nkv != 4:
        raise SystemExit("[k] FATAL: STALE CODE — n_kv_heads != 4, refusing to burn GPU time")
except SystemExit:
    raise
except Exception as e:
    log(f"code stamp failed (continuing): {e}")
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
    log(f"reusing tokens.bin from previous session output "
        f"({os.path.getsize(os.path.join(prev, 'tokens.bin')):,} bytes)")
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
if r.returncode != 0:
    # v5/v6 lesson: exiting 0 here made Kaggle report COMPLETE on crashes
    sys.exit(r.returncode)

# ---------------------------------------------------------------- report
lp = os.path.join(RUN, "latest.pt")
if os.path.exists(lp):
    log(f"checkpoint size: {os.path.getsize(lp):,} bytes")
log(f"working dir contents: {os.listdir(WORK)}")
log(f"session wall time: {(time.time() - t0) / 3600:.2f}h")
log("done — save this version so the next session can chain it")
