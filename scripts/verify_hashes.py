#!/usr/bin/env python3
"""Verify artifact integrity for the RCG-TT release.

Two independent checks:

1. Paper-provenance check (default): every artifact and source artifact recorded
   in evidence/artifact-ledger.jsonl is re-hashed (SHA-256) and compared against
   the digest stored when the paper was built. A match proves the file in this
   repository is byte-identical to the one behind the corresponding table/figure.

2. Release-manifest check (--manifest): re-hash every file listed in
   RELEASE_SHA256SUMS.txt and confirm it matches (a plain integrity check of the
   whole distribution).

Known intentional exception: experiments/derived/aider_rcg/aider_splits.json was
path-normalized for public release (absolute local image paths -> dataset-relative
paths), so its digest intentionally differs from the original study ledger. It is
reported as NORMALIZED rather than FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "evidence" / "artifact-ledger.jsonl"
MANIFEST = REPO / "RELEASE_SHA256SUMS.txt"

# Study-relative prefixes already match this repo's layout, so no remapping is
# needed. Kept explicit for clarity/future-proofing.
NORMALIZED = {"experiments/derived/aider_rcg/aider_splits.json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def check_ledger() -> int:
    if not LEDGER.exists():
        print(f"ERROR: ledger not found at {LEDGER}")
        return 2
    seen: dict[str, str] = {}
    for line in LEDGER.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        entries = [(rec["path"], rec["sha256"])]
        for src in rec.get("source_artifacts", []):
            entries.append((src["path"], src["sha256"]))
        for rel, expected in entries:
            if rel in seen:
                continue  # each file checked once
            seen[rel] = expected

    ok = fail = normalized = missing = 0
    for rel, expected in sorted(seen.items()):
        p = REPO / rel
        if not p.exists():
            print(f"MISSING  {rel}")
            missing += 1
            continue
        actual = sha256(p)
        if actual == expected:
            print(f"OK       {rel}")
            ok += 1
        elif rel in NORMALIZED:
            print(f"NORMAL   {rel}  (path-normalized for release; digest differs by design)")
            normalized += 1
        else:
            print(f"FAIL     {rel}")
            print(f"           expected {expected}")
            print(f"           actual   {actual}")
            fail += 1
    print("-" * 60)
    print(f"ledger check: {ok} ok, {normalized} normalized, {fail} fail, {missing} missing")
    return 1 if (fail or missing) else 0


def check_manifest() -> int:
    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST} (run scripts/make_manifest.py first)")
        return 2
    ok = fail = missing = 0
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, rel = line.split(None, 1)
        p = REPO / rel
        if not p.exists():
            print(f"MISSING  {rel}")
            missing += 1
            continue
        if sha256(p) == expected:
            ok += 1
        else:
            print(f"FAIL     {rel}")
            fail += 1
    print(f"manifest check: {ok} ok, {fail} fail, {missing} missing")
    return 1 if (fail or missing) else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", action="store_true", help="check RELEASE_SHA256SUMS.txt instead of the ledger")
    args = ap.parse_args()
    return check_manifest() if args.manifest else check_ledger()


if __name__ == "__main__":
    sys.exit(main())
