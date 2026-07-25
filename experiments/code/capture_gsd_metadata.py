#!/usr/bin/env python3
"""Capture and hash the fixed CRASAR GSD metadata rows used by the paper."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
REVISION = "47cf4ab3a94d42978975f7d23338a996125ac0e9"
OUTPUT = ROOT / "experiments/derived/metadata"
NAMES = {
    "090403-Lancaster-Canyon-Gate.geo.tif",
    "1001-Summerlin-San-Carlos.geo.tif",
    "1001-Harlem-Heights.geo.tif",
    "1001-McGregor-College-Pkwy-South.1.geo.tif",
    "10132018-MexicoBeach.geo.tif",
    "10142018-MexicoBeach.geo.tif",
    (
        "1001-Harlem-Heights.geo.tif_"
        "10300100DB06A700-visual.tif.geo.tif"
    ),
    (
        "1001-McGregor-College-Pkwy-South.1.geo.tif_"
        "10300100DB06A700-visual.tif.geo.tif"
    ),
    "10132018-MexicoBeach.geo.tif_104001004384D900.tif.geo.tif",
    "10142018-MexicoBeach.geo.tif_104001004384D900.tif.geo.tif",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = Path(
        hf_hub_download(
            "CRASAR/CRASAR-U-DROIDs",
            "statistics.csv",
            repo_type="dataset",
            revision=REVISION,
        )
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "CRASAR_statistics.csv"
    shutil.copy2(source, destination)
    frame = pd.read_csv(destination)
    selected = frame[frame["Orthomosaic"].isin(NAMES)].copy()
    missing = sorted(NAMES - set(selected["Orthomosaic"]))
    if missing:
        raise AssertionError(f"missing metadata rows: {missing}")
    records = selected[
        [
            "Orthomosaic",
            "Train/Test",
            "Source",
            "Platform / Provider",
            "Event",
            "Date (mm/dd/yyy)",
            "Pre/Post Event",
            "GSD (m/px)",
        ]
    ].to_dict(orient="records")
    result = {
        "dataset": "CRASAR/CRASAR-U-DROIDs",
        "revision": REVISION,
        "source_path": "statistics.csv",
        "source_sha256": sha256(destination),
        "selected_rows": records,
    }
    output = OUTPUT / "CRASAR_gsd_records.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("GSD_METADATA_CAPTURED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
