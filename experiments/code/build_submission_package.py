#!/usr/bin/env python3
"""Build deterministic IJRS source and reproducibility archives."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "submission"
FIXED_TIME = (2026, 7, 24, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def include_source(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if not path.is_file() or relative.parts[0] != "paper":
        return False
    return path.suffix.lower() in {
        ".tex",
        ".bib",
        ".cls",
        ".bst",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".eps",
    } and path.name not in {
        "main.aux",
        "main.bbl",
        "main.blg",
        "main.log",
        "main.out",
    }


def include_reproducibility(path: Path) -> bool:
    if not path.is_file():
        return False
    relative = path.relative_to(ROOT)
    if relative.parts[0] in {"submission", "environment"}:
        return False
    if "__pycache__" in relative.parts or path.suffix == ".pyc":
        return False
    if (
        relative.parts[0] == "experiments"
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    ):
        return False
    if relative.as_posix() == "experiments/derived/idalia_paired/pair_manifest.json":
        return False
    if relative.parts[0] in {
        "experiments",
        "evidence",
        "research",
        "paper",
        "reviews",
    }:
        if relative.parts[0] == "paper":
            return include_source(path)
        return True
    return relative.as_posix() in {
        "study.json",
        "LICENSE",
        "LICENSES.md",
        "requirements-lock.txt",
        "reproduce.sh",
        "README.md",
    }


def write_archive(path: Path, files: list[Path]) -> list[dict]:
    manifest = []
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for source in sorted(files):
            relative = source.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            payload = source.read_bytes()
            archive.writestr(info, payload)
            manifest.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
    return manifest


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_archive = OUTPUT / "resolution-audit-source.zip"
    reproducibility_archive = OUTPUT / "resolution-audit-reproducibility.zip"
    source_files = [path for path in (ROOT / "paper").rglob("*") if include_source(path)]
    reproducibility_files = [
        path for path in ROOT.rglob("*") if include_reproducibility(path)
    ]
    source_manifest = write_archive(source_archive, source_files)
    reproducibility_manifest = write_archive(
        reproducibility_archive, reproducibility_files
    )
    checksums = {
        "source_archive": {
            "path": source_archive.name,
            "sha256": sha256(source_archive),
            "bytes": source_archive.stat().st_size,
            "files": len(source_manifest),
        },
        "reproducibility_archive": {
            "path": reproducibility_archive.name,
            "sha256": sha256(reproducibility_archive),
            "bytes": reproducibility_archive.stat().st_size,
            "files": len(reproducibility_manifest),
        },
    }
    (OUTPUT / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n"
    )
    (OUTPUT / "reproducibility_manifest.json").write_text(
        json.dumps(reproducibility_manifest, indent=2) + "\n"
    )
    print("SUBMISSION_PACKAGE_COMPLETE", json.dumps(checksums), flush=True)


if __name__ == "__main__":
    main()
