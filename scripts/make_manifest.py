#!/usr/bin/env python3
"""Generate RELEASE_SHA256SUMS.txt over every tracked file in the repository."""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "RELEASE_SHA256SUMS.txt"
SKIP_NAMES = {"RELEASE_SHA256SUMS.txt", ".DS_Store"}
SKIP_DIRS = {".git", "__pycache__"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    rows = []
    for p in sorted(REPO.rglob("*")):
        if not p.is_file():
            continue
        if p.name in SKIP_NAMES:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(REPO).parts):
            continue
        rel = p.relative_to(REPO).as_posix()
        rows.append(f"{sha256(p)}  {rel}")
    OUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {OUT} ({len(rows)} files)")


if __name__ == "__main__":
    main()
