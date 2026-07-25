#!/usr/bin/env python3
"""Synchronize the approved public GitHub/Zenodo release working tree."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT.parents[1] / "rcg-tt-release"
EXCLUDED_NAMES = {
    ".DS_Store",
    "main.aux",
    "main.bbl",
    "main.blg",
    "main.log",
    "main.out",
    "resolution-audit-reproducibility.zip",
}


def include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if not path.is_file() or path.name in EXCLUDED_NAMES:
        return False
    if "__pycache__" in relative.parts or path.suffix == ".pyc":
        return False
    if relative.parts[0] in {"environment"}:
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not (TARGET / ".git").is_dir():
        raise RuntimeError(f"target is not the expected git checkout: {TARGET}")
    for child in TARGET.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    copied = []
    for source in sorted(ROOT.rglob("*")):
        if not include(source):
            continue
        relative = source.relative_to(ROOT)
        destination = TARGET / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    zenodo = {
        "title": (
            "When Resolution Consistency Fails: Information-Matched Audit "
            "Reproducibility Package"
        ),
        "upload_type": "software",
        "description": (
            "Version 2.0.2 source, checkpoints, per-example scores, logs, "
            "ledgers, and IJRS manuscript for an information-matched audit of "
            "resolution reliability in disaster remote sensing."
        ),
        "creators": [
            {
                "name": "Sharma, Akshay",
                "affiliation": "SAGE University, Indore, India",
            },
            {
                "name": "Prasad, Lalji",
                "affiliation": "SAGE University, Indore, India",
            },
        ],
        "license": "cc-by-4.0",
        "version": "2.0.2",
        "keywords": [
            "remote sensing",
            "resolution shift",
            "uncertainty",
            "data leakage",
            "reproducibility",
        ],
        "related_identifiers": [
            {
                "identifier": "https://github.com/akssha74/rcg-tt",
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        ],
    }
    (TARGET / ".zenodo.json").write_text(json.dumps(zenodo, indent=2) + "\n")
    (TARGET / "CITATION.cff").write_text(
        """cff-version: 1.2.0
message: "If you use this software or evidence package, please cite it."
title: "When Resolution Consistency Fails: Information-Matched Audit Reproducibility Package"
version: 2.0.2
date-released: 2026-07-25
authors:
  - family-names: Sharma
    given-names: Akshay
    affiliation: SAGE University, Indore, India
  - family-names: Prasad
    given-names: Lalji
    affiliation: SAGE University, Indore, India
repository-code: "https://github.com/akssha74/rcg-tt"
doi: "10.5281/zenodo.21510152"
license: CC-BY-4.0
"""
    )
    (TARGET / ".gitignore").write_text(
        "__pycache__/\n*.py[cod]\n.DS_Store\n.venv/\nenvironment/\n"
        "submission/resolution-audit-reproducibility.zip\n"
    )
    release_files = [
        path
        for path in TARGET.rglob("*")
        if path.is_file() and ".git" not in path.parts
    ]
    checksums = [
        f"{sha256(path)}  {path.relative_to(TARGET).as_posix()}"
        for path in sorted(release_files)
        if path.name != "RELEASE_SHA256SUMS.txt"
    ]
    (TARGET / "RELEASE_SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    largest = max(
        (
            (path.stat().st_size, path.relative_to(TARGET).as_posix())
            for path in release_files
        ),
        default=(0, ""),
    )
    print(
        "PUBLIC_RELEASE_SYNCHRONIZED",
        json.dumps(
            {
                "files": len(release_files),
                "largest_file": largest[1],
                "largest_bytes": largest[0],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
