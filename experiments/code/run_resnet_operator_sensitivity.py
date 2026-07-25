#!/usr/bin/env python3
"""Four-operator ResNet sensitivity on deduplicated AIDER and Hurricane."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "imported/experiments/code")
)
import run_primary_multiseed as base

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_mobilenet_operator_replication as helper


ROOT = Path(__file__).resolve().parents[2]
IMPORTED = ROOT / "experiments/imported/experiments/derived/greatness_iteration3"
OUTPUT = ROOT / "experiments/derived/architecture_replication"
SEEDS = (101, 202, 303)
SCALES = (1, 2, 4, 8, 16, 32, 64)


class OperatorItems(Dataset):
    def __init__(self, rows, image_getter, label_getter, scale, operator):
        self.rows = rows
        self.image_getter = image_getter
        self.label_getter = label_getter
        self.scale = scale
        self.operator = operator

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = self.image_getter(row).convert("RGB")
        image = helper.degrade(image, self.scale, self.operator)
        return base.EVAL_TF(image), int(self.label_getter(row))


def corpus_data(corpus):
    if corpus == "aider":
        return base.aider_data(IMPORTED / "aider_splits_dedup.json")
    return base.hurricane_data(IMPORTED / "hurricane_splits_dedup.json")


def evaluate_seed(corpus, seed, device):
    (
        _,
        _,
        test_rows,
        image_getter,
        label_getter,
        _,
        _,
        num_labels,
    ) = corpus_data(corpus)
    checkpoint = IMPORTED / f"{corpus}/seed_{seed}/best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = base.build_model(num_labels)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    output = {}
    score_root = OUTPUT / "resnet_scores" / corpus / f"seed_{seed}"
    score_root.mkdir(parents=True, exist_ok=True)
    for operator_index, operator in enumerate(helper.OPERATORS):
        ladder = {}
        labels = None
        for scale in SCALES:
            ladder[scale], labels, _ = base.predict(
                model,
                OperatorItems(
                    test_rows, image_getter, label_getter, scale, operator
                ),
                device,
            )
        prediction = ladder[8].argmax(1)
        errors = (prediction != labels).astype(int)
        received = np.mean(
            [helper.js(ladder[8], ladder[scale]) for scale in (16, 32, 64)],
            axis=0,
        )
        fine = np.mean(
            [helper.js(ladder[8], ladder[scale]) for scale in (1, 2, 4)],
            axis=0,
        )
        comparison = helper.bootstrap_difference(
            errors,
            fine,
            received,
            seed=550_000 + seed + operator_index * 10_000,
        )
        score_path = score_root / f"{operator}.npz"
        np.savez_compressed(
            score_path,
            labels=labels.astype(np.int64),
            predictions=prediction.astype(np.int64),
            errors=errors.astype(np.int8),
            received_consistency=received,
            fine_reference_consistency=fine,
        )
        output[operator] = {
            "n": len(labels),
            "n_errors": int(errors.sum()),
            "accuracy": float((prediction == labels).mean()),
            "received_consistency": helper.error_metrics(errors, received),
            "fine_reference_consistency": helper.error_metrics(errors, fine),
            "fine_minus_received": comparison,
            "score_path": str(score_path.relative_to(ROOT)),
        }
        print(
            "RESNET_OPERATOR_COMPLETE",
            corpus,
            seed,
            operator,
            json.dumps(output[operator]),
            flush=True,
        )
    return output


def aggregate(results):
    output = {}
    all_pass = True
    for corpus in ("aider", "hurricane"):
        output[corpus] = {}
        for operator in helper.OPERATORS:
            values = np.asarray(
                [
                    results[corpus][str(seed)][operator]["fine_minus_received"][
                        "mean"
                    ]
                    for seed in SEEDS
                ]
            )
            lower = [
                results[corpus][str(seed)][operator]["fine_minus_received"][
                    "ci95"
                ][0]
                for seed in SEEDS
            ]
            row = {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "positive_seeds": int(np.sum(values > 0)),
                "positive_interval_seeds": sum(value > 0 for value in lower),
            }
            row["pass"] = (
                row["mean"] > 0
                and row["positive_seeds"] == 3
                and row["positive_interval_seeds"] >= 2
            )
            all_pass = all_pass and row["pass"]
            output[corpus][operator] = row
    output["m3_cross_architecture_operator_pass"] = all_pass
    return output


def main():
    started = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = {}
    for corpus in ("aider", "hurricane"):
        results[corpus] = {}
        for seed in SEEDS:
            results[corpus][str(seed)] = evaluate_seed(
                corpus, seed, device
            )
    summary = {
        "protocol": {
            "corpora": ["aider", "hurricane"],
            "architecture": "transformers microsoft/resnet-18",
            "seeds": list(SEEDS),
            "operators": list(helper.OPERATORS),
            "scales": list(SCALES),
            "bootstrap_repeats": helper.BOOTSTRAP_REPEATS,
            "analysis_status": "prospective operator sensitivity",
            "device": str(device),
        },
        "results": results,
        "aggregate": aggregate(results),
        "runtime_seconds": time.time() - started,
    }
    output = OUTPUT / "resnet_operator_sensitivity.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("RESNET_OPERATOR_SENSITIVITY_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
