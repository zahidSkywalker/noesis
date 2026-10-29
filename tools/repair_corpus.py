"""Post-build corpus repair: validate every .txt.gz shard, salvage truncated ones.

A shard written when the builder was killed mid-write has no gzip trailer and
crashes naive consumers. For each invalid shard we salvage the decompressable
prefix and rewrite the file as a valid gz. The loss (tail of one partial
document) is negligible against the ~7B-char corpus.

Run AFTER corpus build completes, BEFORE packaging/uploading to Kaggle.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys

CORPUS = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/corpus"


def inspect(path: str) -> tuple[bool, bytes]:
    """Returns (valid, payload). payload = full body if valid else salvaged prefix."""
    data = bytearray()
    try:
        with gzip.open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                data += chunk
        return True, bytes(data)
    except EOFError:
        return False, bytes(data)


def main():
    shards = sorted(glob.glob(os.path.join(CORPUS, "*.txt.gz")))
    print(f"[repair] validating {len(shards)} shard(s) in {CORPUS}")
    bad = []
    report = []
    for p in shards:
        ok, payload = inspect(p)
        if not ok:
            bad.append(p)
            tmp = p + ".fix"
            with gzip.open(tmp, "wb", compresslevel=6) as f:
                f.write(payload)
            os.replace(tmp, p)
            ok2, payload2 = inspect(p)
            report.append({"shard": os.path.basename(p), "status": "salvaged",
                           "bytes": len(payload2), "revalidated": ok2})
            print(f"[repair] SALVAGED {os.path.basename(p)}: {len(payload2):,} bytes "
                  f"prefix ({'valid' if ok2 else 'STILL BROKEN'})")
        else:
            report.append({"shard": os.path.basename(p), "status": "ok", "bytes": len(payload)})

    n_ok = sum(1 for r in report if r["status"] == "ok")
    print(f"[repair] done: {n_ok}/{len(shards)} were valid, {len(bad)} salvaged")
    with open(os.path.join(CORPUS, "repair_report.json"), "w") as f:
        json.dump({"checked": len(shards), "salvaged": [r["shard"] for r in report
                                                             if r["status"] == "salvaged"],
                   "details": report}, f, indent=2)
    if any(not r["revalidated"] for r in report if r["status"] == "salvaged"):
        print("[repair] FAILED: a salvaged shard did not revalidate")
        sys.exit(2)


if __name__ == "__main__":
    main()
