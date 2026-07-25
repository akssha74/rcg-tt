#!/usr/bin/env python3
"""Three-seed MobileNet and four-operator oracle/received audit replication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = (
    ROOT
    / "experiments/imported/experiments/derived/greatness_iteration3/"
    "aider_splits_dedup.json"
)
OUTPUT = ROOT / "experiments/derived/architecture_replication"
SEEDS = (101, 202, 303)
SCALES = (1, 2, 4, 8, 16, 32, 64)
OPERATORS = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
    "box": Image.Resampling.BOX,
}
BOOTSTRAP_REPEATS = 2_000

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = os.environ.get("AIDER_ROOT")
    if not root:
        raise RuntimeError("AIDER_ROOT is required for relative AIDER paths")
    return Path(root) / path


def degrade(image: Image.Image, scale: int, operator: str) -> Image.Image:
    if scale == 1:
        return image
    width, height = image.size
    resampling = OPERATORS[operator]
    reduced = image.resize(
        (max(1, width // scale), max(1, height // scale)), resampling
    )
    return reduced.resize((width, height), resampling)


class Images(Dataset):
    def __init__(self, rows, train=False, scale=1, operator="bicubic"):
        self.rows = rows
        self.transform = TRAIN_TRANSFORM if train else EVAL_TRANSFORM
        self.scale = scale
        self.operator = operator

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(resolve_path(row["path"])) as image:
            image = degrade(image.convert("RGB"), self.scale, self.operator)
            tensor = self.transform(image)
        return tensor, int(row["label"])


def build_model() -> nn.Module:
    return timm.create_model(
        "mobilenetv3_small_100",
        pretrained=True,
        num_classes=5,
    )


@torch.no_grad()
def predict(model, dataset, device):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    probabilities, labels = [], []
    model.eval()
    for images, target in loader:
        logits = model(images.to(device))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(target.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def class_weights(rows):
    labels = np.asarray([int(row["label"]) for row in rows])
    counts = np.bincount(labels, minlength=5).astype(float)
    return torch.tensor(len(labels) / (5 * counts), dtype=torch.float32)


def train(seed, train_rows, val_rows, device):
    set_seed(seed)
    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows).to(device))
    loader = DataLoader(
        Images(train_rows, train=True),
        batch_size=64,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    seed_root = OUTPUT / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    checkpoint = seed_root / "best.pt"
    best = -math.inf
    history = []
    for epoch in range(1, 7):
        model.train()
        loss_sum = 0.0
        count = 0
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(target)
            count += len(target)
        probability, labels = predict(model, Images(val_rows), device)
        macro_f1 = float(f1_score(labels, probability.argmax(1), average="macro"))
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "val_macro_f1": macro_f1,
        }
        history.append(row)
        print("MOBILENET_EPOCH", seed, json.dumps(row), flush=True)
        if macro_f1 > best:
            best = macro_f1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "val_macro_f1": macro_f1,
                },
                checkpoint,
            )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    (seed_root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, best, checkpoint


def js(p, q, eps=1e-9):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    midpoint = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / midpoint), axis=1) + 0.5 * np.sum(
        q * np.log(q / midpoint), axis=1
    )


def safe_auc(labels, score):
    return float(roc_auc_score(labels, score))


def error_metrics(errors, score):
    fpr, tpr, _ = roc_curve(errors, score)
    eligible = np.where(tpr >= 0.95)[0]
    return {
        "auroc": safe_auc(errors, score),
        "auprc": float(average_precision_score(errors, score)),
        "fpr95": float(fpr[eligible[0]]) if len(eligible) else 1.0,
    }


def bootstrap_difference(errors, first, second, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_REPEATS):
        indexes = rng.integers(0, len(errors), len(errors))
        if len(np.unique(errors[indexes])) < 2:
            continue
        values.append(
            roc_auc_score(errors[indexes], first[indexes])
            - roc_auc_score(errors[indexes], second[indexes])
        )
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ],
        "repeats": len(values),
    }


def evaluate(model, rows, seed, device):
    output = {}
    score_root = OUTPUT / f"seed_{seed}/scores"
    score_root.mkdir(parents=True, exist_ok=True)
    for operator_index, operator in enumerate(OPERATORS):
        ladder = {}
        labels = None
        for scale in SCALES:
            ladder[scale], labels = predict(
                model, Images(rows, scale=scale, operator=operator), device
            )
        prediction = ladder[8].argmax(1)
        errors = (prediction != labels).astype(int)
        confidence = 1.0 - ladder[8].max(1)
        reference = np.mean([js(ladder[1], ladder[s]) for s in (2, 4, 8)], axis=0)
        received = np.mean([js(ladder[8], ladder[s]) for s in (16, 32, 64)], axis=0)
        fine_anchor = np.mean([js(ladder[8], ladder[s]) for s in (1, 2, 4)], axis=0)
        comparison = bootstrap_difference(
            errors,
            fine_anchor,
            received,
            seed=350_000 + seed + operator_index * 10_000,
        )
        reference_comparison = bootstrap_difference(
            errors,
            reference,
            received,
            seed=450_000 + seed + operator_index * 10_000,
        )
        score_path = score_root / f"{operator}.npz"
        np.savez_compressed(
            score_path,
            labels=labels.astype(np.int64),
            predictions=prediction.astype(np.int64),
            errors=errors.astype(np.int8),
            confidence=confidence,
            reference_consistency=reference,
            received_consistency=received,
            fine_reference_consistency=fine_anchor,
        )
        output[operator] = {
            "n": len(labels),
            "n_errors": int(errors.sum()),
            "accuracy": float((prediction == labels).mean()),
            "confidence": error_metrics(errors, confidence),
            "reference_consistency": error_metrics(errors, reference),
            "received_consistency": error_metrics(errors, received),
            "fine_reference_consistency": error_metrics(errors, fine_anchor),
            "fine_minus_received": comparison,
            "reference_minus_received": reference_comparison,
            "score_path": str(score_path.relative_to(ROOT)),
        }
        print(
            "MOBILENET_OPERATOR_COMPLETE",
            seed,
            operator,
            json.dumps(output[operator]),
            flush=True,
        )
    return output


def aggregate(seed_results):
    aggregate = {"operators": {}}
    for operator in OPERATORS:
        gaps = np.asarray(
            [
                seed_results[str(seed)]["operators"][operator][
                    "fine_minus_received"
                ]["mean"]
                for seed in SEEDS
            ]
        )
        lowers = [
            seed_results[str(seed)]["operators"][operator][
                "fine_minus_received"
            ]["ci95"][0]
            for seed in SEEDS
        ]
        aggregate["operators"][operator] = {
            "fine_minus_received_values": gaps.tolist(),
            "mean": float(gaps.mean()),
            "std": float(gaps.std(ddof=1)),
            "positive_seeds": int(np.sum(gaps > 0)),
            "positive_interval_seeds": sum(value > 0 for value in lowers),
        }
    aggregate["m1_architecture_pass"] = (
        aggregate["operators"]["bicubic"]["positive_seeds"] == 3
        and aggregate["operators"]["bicubic"]["positive_interval_seeds"] == 3
    )
    aggregate["m2_operator_pass"] = all(
        values["mean"] > 0
        and values["positive_seeds"] == 3
        and values["positive_interval_seeds"] >= 2
        for values in aggregate["operators"].values()
    )
    return aggregate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.time()
    split = json.loads(SPLIT_PATH.read_text())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for seed in SEEDS:
        model, best, checkpoint = train(
            seed, split["train"], split["val"], device
        )
        results[str(seed)] = {
            "seed": seed,
            "best_val_macro_f1": best,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(checkpoint),
            "operators": evaluate(model, split["test"], seed, device),
        }
    summary = {
        "protocol": {
            "dataset": "AIDER v1.0 exact-hash-clean split",
            "architecture": "timm mobilenetv3_small_100 ImageNet-1K",
            "seeds": list(SEEDS),
            "epochs": 6,
            "operators": list(OPERATORS),
            "scales": list(SCALES),
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "device": str(device),
        },
        "seeds": results,
        "aggregate": aggregate(results),
        "runtime_seconds": time.time() - started,
    }
    output = OUTPUT / "mobilenet_aider_operator_audit.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("MOBILENET_REPLICATION_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
