#!/usr/bin/env python3
"""Multi-site paired same-building measured-GSD replication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_paired_measured_gsd as paired
import run_measured_gsd_crasar as measured
import run_primary_multiseed as base


ROOT = Path(__file__).resolve().parents[2]
CRASAR = ROOT / "experiments/derived/greatness_strengthening/crasar"
REPO = "CRASAR/CRASAR-U-DROIDs"
SNAPSHOT = paired.SNAPSHOT
VALID = paired.VALID
SITES = [
    {
        "id": "harlem-heights",
        "uas_name": "1001-Harlem-Heights.geo.tif",
        "sat_name": (
            "1001-Harlem-Heights.geo.tif_"
            "10300100DB06A700-visual.tif.geo.tif"
        ),
        "uas_gsd_m": 0.046720,
        "satellite_gsd_m": 0.305175781,
        "event": "Hurricane Ian",
    },
    {
        "id": "mcgregor-college-parkway-south-1",
        "uas_name": "1001-McGregor-College-Pkwy-South.1.geo.tif",
        "sat_name": (
            "1001-McGregor-College-Pkwy-South.1.geo.tif_"
            "10300100DB06A700-visual.tif.geo.tif"
        ),
        "uas_gsd_m": 0.0383888,
        "satellite_gsd_m": 0.305175781,
        "event": "Hurricane Ian",
    },
]


def prepare_site(site: dict, force: bool = False) -> list[dict]:
    output = CRASAR / "multisite_paired" / site["id"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "pair_manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    uas_manifest = json.loads((CRASAR / "patch_manifest.json").read_text())
    uas_rows = {
        row["building_id"]: row
        for row in uas_manifest
        if row["orthomosaic"] == site["uas_name"]
        and row["split"] in {"val", "test"}
    }
    sat_image = Path(
        hf_hub_download(
            REPO,
            f"train/imagery/SATELLITE/{site['sat_name']}",
            repo_type="dataset",
            revision=SNAPSHOT,
        )
    )
    sat_annotation = Path(
        hf_hub_download(
            REPO,
            "train/annotations/SATELLITE/building_damage_assessment/"
            f"{site['sat_name']}.json",
            repo_type="dataset",
            revision=SNAPSHOT,
        )
    )
    uas_annotation = Path(
        hf_hub_download(
            REPO,
            "train/annotations/UAS/building_damage_assessment/"
            f"{site['uas_name']}.json",
            repo_type="dataset",
            revision=SNAPSHOT,
        )
    )
    sat = {row["building_id"]: row for row in json.loads(sat_annotation.read_text())}
    uas = {row["building_id"]: row for row in json.loads(uas_annotation.read_text())}
    if not uas_rows:
        uas_image = Path(
            hf_hub_download(
                REPO,
                f"train/imagery/UAS/{site['uas_name']}",
                repo_type="dataset",
                revision=SNAPSHOT,
            )
        )
        uas_patch_dir = output / "uas_patches"
        uas_patch_dir.mkdir(parents=True, exist_ok=True)
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(uas_image) as raster:
            for building_id, item in uas.items():
                if item.get("label") not in VALID:
                    continue
                pixels = item.get("pixels") or []
                if not pixels:
                    continue
                centre_x = float(np.mean([point["x"] for point in pixels]))
                centre_y = float(np.mean([point["y"] for point in pixels]))
                split = measured.spatial_split(
                    site["uas_name"], centre_x, centre_y
                )
                if split not in {"val", "test"}:
                    continue
                patch_path = uas_patch_dir / f"{building_id}.jpg"
                if force or not patch_path.exists():
                    half = 256
                    patch = raster.crop(
                        (
                            int(round(centre_x)) - half,
                            int(round(centre_y)) - half,
                            int(round(centre_x)) + half,
                            int(round(centre_y)) + half,
                        )
                    ).convert("RGB")
                    patch.save(patch_path, quality=92, optimize=True)
                uas_rows[building_id] = {
                    "building_id": building_id,
                    "label": measured.LABELS[item["label"]],
                    "source_label": item["label"],
                    "split": split,
                    "path": str(patch_path),
                }
    patch_dir = output / "satellite_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(sat_image) as raster:
        for building_id, uas_row in uas_rows.items():
            if building_id not in sat or building_id not in uas:
                continue
            sat_item, uas_item = sat[building_id], uas[building_id]
            if (
                sat_item.get("label") not in VALID
                or sat_item.get("label") != uas_item.get("label")
            ):
                continue
            pixels = sat_item.get("pixels") or []
            if not pixels:
                continue
            centre_x = float(np.mean([point["x"] for point in pixels]))
            centre_y = float(np.mean([point["y"] for point in pixels]))
            patch_path = patch_dir / f"{building_id}.jpg"
            if force or not patch_path.exists():
                half = 256
                patch = raster.crop(
                    (
                        int(round(centre_x)) - half,
                        int(round(centre_y)) - half,
                        int(round(centre_x)) + half,
                        int(round(centre_y)) + half,
                    )
                ).convert("RGB")
                patch.save(patch_path, quality=92, optimize=True)
            rows.append(
                {
                    "site": site["id"],
                    "event": site["event"],
                    "building_id": building_id,
                    "label": uas_row["label"],
                    "source_label": uas_item["label"],
                    "split": uas_row["split"],
                    "uas_path": uas_row["path"],
                    "satellite_path": str(patch_path),
                    "uas_gsd_m": site["uas_gsd_m"],
                    "satellite_gsd_m": site["satellite_gsd_m"],
                }
            )
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def summarize_modality(seed_results: dict[str, dict], scope: str, modality: str):
    out = {}
    for metric in (
        "accuracy",
        "error_confidence",
        "auroc_confidence",
        "auroc_rcg",
        "lift_rcg_minus_confidence",
    ):
        values = np.asarray(
            [
                seed_results[str(seed)][scope][modality][metric]
                for seed in base.SEEDS
            ],
            dtype=float,
        )
        out[metric] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return out


def main() -> None:
    site_rows = {site["id"]: prepare_site(site) for site in SITES}
    counts = {
        site_id: {
            split: {
                "n": sum(row["split"] == split for row in rows),
                "undamaged": sum(
                    row["split"] == split and row["label"] == 0 for row in rows
                ),
                "damaged": sum(
                    row["split"] == split and row["label"] == 1 for row in rows
                ),
            }
            for split in ("val", "test")
        }
        for site_id, rows in site_rows.items()
    }
    print("MULTISITE_COUNTS", json.dumps(counts), flush=True)
    pooled_rows = [row for rows in site_rows.values() for row in rows]
    scopes = {**site_rows, "pooled": pooled_rows}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    seed_results = {}
    for seed in base.SEEDS:
        state = torch.load(
            CRASAR / f"seed_{seed}/best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = base.build_model(2)
        model.load_state_dict(state["state_dict"])
        model.to(device)
        result = {"seed": seed}
        for scope, rows in scopes.items():
            test_rows = [row for row in rows if row["split"] == "test"]
            uas = paired.modality_scores(model, test_rows, "uas_path", device)
            satellite = paired.modality_scores(
                model, test_rows, "satellite_path", device
            )
            result[scope] = {
                "uas": uas["metrics"],
                "satellite": satellite["metrics"],
            }
        seed_results[str(seed)] = result
        print("MULTISITE_SEED_COMPLETE", seed, json.dumps(result), flush=True)

    aggregate = {}
    for scope in scopes:
        aggregate[scope] = {
            "uas": summarize_modality(seed_results, scope, "uas"),
            "satellite": summarize_modality(
                seed_results, scope, "satellite"
            ),
        }
        aggregate[scope]["accuracy_drop"] = (
            aggregate[scope]["uas"]["accuracy"]["mean"]
            - aggregate[scope]["satellite"]["accuracy"]["mean"]
        )
        aggregate[scope]["satellite_rcg_lift"] = aggregate[scope][
            "satellite"
        ]["lift_rcg_minus_confidence"]["mean"]
    aggregate["w6c_pass"] = (
        aggregate["pooled"]["accuracy_drop"] > 0
        and aggregate["pooled"]["satellite_rcg_lift"] > 0
        and all(aggregate[site["id"]]["accuracy_drop"] > 0 for site in SITES)
    )
    summary = {
        "protocol": {
            "dataset": REPO,
            "snapshot": SNAPSHOT,
            "sites": SITES,
            "same_label_required": True,
            "seeds": list(base.SEEDS),
        },
        "pair_counts": counts,
        "seeds": seed_results,
        "aggregate": aggregate,
    }
    output = CRASAR / "multisite_paired_gsd.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("MULTISITE_GSD_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
