#!/usr/bin/env python3
"""Verify archive manifests and clean-build the IJRS source package."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "submission"

# Tectonic stamps the build time into the PDF unless the date is forced, which
# makes the recorded pdf_sha256 differ on every rerun and invalidates the hash
# registered for this file in the run and artifact ledgers.
BUILD_EPOCH = "1700000000"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_manifest(archive_path: Path, manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    expected = {row["path"]: row for row in manifest}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        if names != set(expected):
            raise AssertionError(
                {
                    "missing": sorted(set(expected) - names),
                    "extra": sorted(names - set(expected)),
                }
            )
        for name in sorted(names):
            if name.startswith("/") or ".." in Path(name).parts:
                raise AssertionError(f"unsafe archive path: {name}")
            payload = archive.read(name)
            row = expected[name]
            if len(payload) != row["bytes"]:
                raise AssertionError(f"size mismatch: {name}")
            if sha256_bytes(payload) != row["sha256"]:
                raise AssertionError(f"hash mismatch: {name}")
    return len(expected)


def main() -> None:
    source_archive = SUBMISSION / "resolution-audit-source.zip"
    reproducibility_archive = (
        SUBMISSION / "resolution-audit-reproducibility.zip"
    )
    source_count = verify_manifest(
        source_archive, SUBMISSION / "source_manifest.json"
    )
    reproducibility_count = verify_manifest(
        reproducibility_archive,
        SUBMISSION / "reproducibility_manifest.json",
    )
    with tempfile.TemporaryDirectory(prefix="resolution-audit-source-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(source_archive) as archive:
            archive.extractall(root)
        environment = dict(os.environ)
        environment["SOURCE_DATE_EPOCH"] = BUILD_EPOCH
        environment["FORCE_SOURCE_DATE"] = "1"
        completed = subprocess.run(
            [
                "tectonic",
                "main.tex",
                "--keep-logs",
                "--keep-intermediates",
            ],
            cwd=root / "paper",
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout + "\n" + completed.stderr)
        built_pdf = root / "paper/main.pdf"
        if not built_pdf.is_file() or built_pdf.stat().st_size == 0:
            raise AssertionError("clean build did not produce paper/main.pdf")
        build = {
            "exit_code": completed.returncode,
            "source_date_epoch": BUILD_EPOCH,
            "pdf_sha256": sha256_bytes(built_pdf.read_bytes()),
            "pdf_bytes": built_pdf.stat().st_size,
        }
    result = {
        "source_manifest_files": source_count,
        "reproducibility_manifest_files": reproducibility_count,
        "clean_source_build": build,
        "all_pass": True,
    }
    output = SUBMISSION / "verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("SUBMISSION_PACKAGE_VERIFIED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
