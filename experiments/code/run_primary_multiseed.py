#!/usr/bin/env python3
"""Run preregistered multi-seed ResNet, RCG, and EO-kNN experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ResNetForImageClassification


ROOT = Path(__file__).resolve().parents[2]
PRETRAINED = Path(
    "/Users/akshay.sharma/.cache/huggingface/hub/"
    "models--microsoft--resnet-18/snapshots/"
    "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
)
SCALES = (1, 2, 4, 8)
SEEDS = (101, 202, 303)
GAMMA = 0.7

TRAIN_TF = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
EVAL_TF = transforms.Compose(
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


def degrade(image: Image.Image, scale: int) -> Image.Image:
    if scale == 1:
        return image
    width, height = image.size
    reduced = image.resize(
        (max(1, width // scale), max(1, height // scale)),
        Image.Resampling.BICUBIC,
    )
    return reduced.resize((width, height), Image.Resampling.BICUBIC)


class ImageItems(Dataset):
    def __init__(
        self,
        items: list,
        image_getter: Callable,
        label_getter: Callable,
        train: bool,
        scale: int = 1,
    ):
        self.items = items
        self.image_getter = image_getter
        self.label_getter = label_getter
        self.transform = TRAIN_TF if train else EVAL_TF
        self.scale = scale

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = self.image_getter(item).convert("RGB")
        image = degrade(image, self.scale)
        return self.transform(image), int(self.label_getter(item))


def build_model(num_labels: int) -> ResNetForImageClassification:
    return ResNetForImageClassification.from_pretrained(
        PRETRAINED,
        local_files_only=True,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )


def logits_and_features(model, batch):
    outputs = model.resnet(pixel_values=batch)
    pooled = outputs.pooler_output
    logits = model.classifier(pooled)
    return logits, pooled.flatten(1)


@torch.no_grad()
def predict(model, dataset: Dataset, device, batch_size: int = 64):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities, labels, features = [], [], []
    model.eval()
    for images, target in loader:
        images = images.to(device)
        logits, embedding = logits_and_features(model, images)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        features.append(embedding.cpu().numpy())
        labels.append(target.numpy())
    return (
        np.concatenate(probabilities),
        np.concatenate(labels),
        np.concatenate(features),
    )


def macro_f1(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, probabilities.argmax(1), average="macro"))


def class_weights(labels: list[int], num_labels: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels), minlength=num_labels).astype(float)
    weights = len(labels) / (num_labels * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def train_one(
    *,
    seed: int,
    num_labels: int,
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_labels: list[int],
    device,
    epochs: int,
    output_dir: Path,
) -> tuple[ResNetForImageClassification, list[dict], float]:
    set_seed(seed)
    model = build_model(num_labels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_labels, num_labels).to(device)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best.pt"
    best_f1 = -math.inf
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pixel_values=images).logits
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(target)
            seen += len(target)
        val_probability, val_labels, _ = predict(model, val_dataset, device)
        value = macro_f1(val_probability, val_labels)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            "val_macro_f1": value,
        }
        history.append(row)
        print(
            f"seed={seed} epoch={epoch}/{epochs} "
            f"loss={row['train_loss']:.5f} val_f1={value:.5f}",
            flush=True,
        )
        if value > best_f1:
            best_f1 = value
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "val_macro_f1": value,
                    "num_labels": num_labels,
                },
                checkpoint,
            )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.to(device)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return model, history, best_f1


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    midpoint = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / midpoint), axis=1) + 0.5 * np.sum(
        q * np.log(q / midpoint), axis=1
    )


def rcg_score(ladder: dict[int, np.ndarray]) -> np.ndarray:
    native = ladder[1]
    values = [js_divergence(native, ladder[scale]) for scale in SCALES[1:]]
    return np.mean(values, axis=0)


def safe_auroc(labels: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, score))


def bootstrap_difference(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    seed: int,
    repeats: int = 1000,
) -> dict:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        index = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[index])) < 2:
            continue
        values.append(
            roc_auc_score(labels[index], score_a[index])
            - roc_auc_score(labels[index], score_b[index])
        )
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "ci95": [
            float(np.percentile(array, 2.5)),
            float(np.percentile(array, 97.5)),
        ],
    }


def normalized(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float64)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def knn_distance(
    train_features: np.ndarray,
    test_features: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    bank = torch.from_numpy(normalized(train_features))
    query = torch.from_numpy(normalized(test_features))
    scores = []
    for start in range(0, len(query), 256):
        similarity = query[start : start + 256] @ bank.T
        nearest = torch.topk(similarity, k=k, dim=1, largest=True).values
        scores.append((1.0 - nearest.mean(dim=1)).numpy())
    output = np.concatenate(scores)
    if not np.all(np.isfinite(output)):
        raise ValueError("non-finite kNN distance")
    return output


def transfer_metrics(
    *,
    val_confidence: np.ndarray,
    val_rcg: np.ndarray,
    test_confidence: np.ndarray,
    test_rcg: np.ndarray,
    test_pred: np.ndarray,
    test_labels: np.ndarray,
    false_critical: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict:
    confidence_threshold = float(np.quantile(val_confidence, 1.0 - GAMMA))
    rcg_threshold = float(np.quantile(val_rcg, GAMMA))
    keep_confidence = test_confidence >= confidence_threshold
    keep_rcg = test_rcg <= rcg_threshold
    false_critical_mask = false_critical(test_labels, test_pred)

    def summarize(mask: np.ndarray) -> dict:
        return {
            "coverage": float(mask.mean()),
            "fcr": (
                float(false_critical_mask[mask].mean()) if np.any(mask) else None
            ),
            "n": int(mask.sum()),
        }

    return {
        "threshold_confidence": confidence_threshold,
        "threshold_rcg": rcg_threshold,
        "confidence": summarize(keep_confidence),
        "rcg": summarize(keep_rcg),
    }


def evaluate_seed(
    *,
    model,
    seed: int,
    train_items: list,
    val_items: list,
    test_items: list,
    image_getter: Callable,
    label_getter: Callable,
    false_critical: Callable[[np.ndarray, np.ndarray], np.ndarray],
    device,
    output_dir: Path,
) -> dict:
    probabilities: dict[int, np.ndarray] = {}
    features: dict[int, np.ndarray] = {}
    labels = None
    for scale in SCALES:
        dataset = ImageItems(
            test_items, image_getter, label_getter, train=False, scale=scale
        )
        prob, current_labels, feat = predict(model, dataset, device)
        probabilities[scale] = prob
        features[scale] = feat
        labels = current_labels
    assert labels is not None
    score_rcg = rcg_score(probabilities)
    confidence_s8 = probabilities[8].max(1)
    prediction_s8 = probabilities[8].argmax(1)
    errors = (prediction_s8 != labels).astype(int)

    train_dataset = ImageItems(
        train_items, image_getter, label_getter, train=False, scale=1
    )
    _, _, train_features = predict(model, train_dataset, device)
    score_knn = knn_distance(train_features, features[8], k=10)
    score_confidence = 1.0 - confidence_s8

    val_probabilities = {}
    for scale in SCALES:
        dataset = ImageItems(
            val_items, image_getter, label_getter, train=False, scale=scale
        )
        prob, _, _ = predict(model, dataset, device)
        val_probabilities[scale] = prob
    val_rcg = rcg_score(val_probabilities)
    val_confidence = val_probabilities[1].max(1)

    ladder = {}
    for scale in SCALES:
        prediction = probabilities[scale].argmax(1)
        confidence = probabilities[scale].max(1)
        wrong = prediction != labels
        ladder[str(scale)] = {
            "accuracy": float((~wrong).mean()),
            "error_confidence": (
                float(confidence[wrong].mean()) if np.any(wrong) else None
            ),
            "n_errors": int(wrong.sum()),
        }

    transfer = transfer_metrics(
        val_confidence=val_confidence,
        val_rcg=val_rcg,
        test_confidence=confidence_s8,
        test_rcg=score_rcg,
        test_pred=prediction_s8,
        test_labels=labels,
        false_critical=false_critical,
    )
    result = {
        "seed": seed,
        "ladder": ladder,
        "s8": {
            "auroc_confidence": safe_auroc(errors, score_confidence),
            "auroc_rcg": safe_auroc(errors, score_rcg),
            "auroc_knn": safe_auroc(errors, score_knn),
        },
        "transfer_s8": transfer,
    }
    result["s8"]["lift_rcg_minus_confidence"] = (
        result["s8"]["auroc_rcg"] - result["s8"]["auroc_confidence"]
    )
    result["s8"]["lift_rcg_minus_knn"] = (
        result["s8"]["auroc_rcg"] - result["s8"]["auroc_knn"]
    )
    result["s8"]["bootstrap_rcg_minus_confidence"] = bootstrap_difference(
        errors, score_rcg, score_confidence, seed=seed
    )
    result["s8"]["bootstrap_rcg_minus_knn"] = bootstrap_difference(
        errors, score_rcg, score_knn, seed=seed + 10_000
    )
    np.savez_compressed(
        output_dir / "test_scores.npz",
        labels=labels,
        errors=errors,
        confidence=score_confidence,
        rcg=score_rcg,
        knn=score_knn,
        prediction=prediction_s8,
    )
    return result


def aggregate(seed_results: dict[str, dict]) -> dict:
    metrics = [
        "auroc_confidence",
        "auroc_rcg",
        "auroc_knn",
        "lift_rcg_minus_confidence",
        "lift_rcg_minus_knn",
    ]
    out = {}
    for metric in metrics:
        values = np.asarray(
            [seed_results[str(seed)]["s8"][metric] for seed in SEEDS], dtype=float
        )
        out[metric] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    out["w5_seeds_over_005"] = int(
        sum(
            seed_results[str(seed)]["s8"]["lift_rcg_minus_confidence"] > 0.05
            for seed in SEEDS
        )
    )
    out["w5_pass"] = (
        out["lift_rcg_minus_confidence"]["mean"] > 0
        and out["w5_seeds_over_005"] >= 2
    )
    out["w7_pass"] = out["lift_rcg_minus_knn"]["mean"] > 0
    return out


def aider_data():
    splits = json.loads(
        (ROOT / "experiments/derived/aider_rcg/aider_splits.json").read_text()
    )
    aider_root_value = os.environ.get("AIDER_ROOT")
    aider_root = Path(aider_root_value) if aider_root_value else None

    def getter(item):
        path = Path(item["path"])
        if not path.is_absolute():
            if aider_root is None:
                raise RuntimeError(
                    "AIDER_ROOT is required for release-normalized split paths"
                )
            path = aider_root / path
        return Image.open(path)
    label = lambda item: item["label"]
    false_critical = lambda y, pred: (y == 4) & (pred != 4)
    labels = [int(item["label"]) for item in splits["train"]]
    return (
        splits["train"],
        splits["val"],
        splits["test"],
        getter,
        label,
        false_critical,
        labels,
        5,
    )


def hurricane_data():
    dataset = load_dataset(
        "jonathan-roberts1/Satellite-Images-of-Hurricane-Damage", split="train"
    )
    order = np.random.default_rng(0).permutation(len(dataset)).tolist()
    train, val, test = order[:7000], order[7000:8000], order[8000:]
    getter = lambda index: dataset[int(index)]["image"]
    label = lambda index: dataset[int(index)]["label"]
    false_critical = lambda y, pred: (y == 1) & (pred == 0)
    labels = [int(dataset[int(index)]["label"]) for index in train]
    return train, val, test, getter, label, false_critical, labels, 2


def run_dataset(
    name: str, epochs: int, device, output_root: Path, eval_only: bool = False
) -> dict:
    data = aider_data() if name == "aider" else hurricane_data()
    (
        train_items,
        val_items,
        test_items,
        image_getter,
        label_getter,
        false_critical,
        train_labels,
        num_labels,
    ) = data
    seed_results = {}
    for seed in SEEDS:
        started = time.time()
        seed_dir = output_root / name / f"seed_{seed}"
        if eval_only:
            state = torch.load(
                seed_dir / "best.pt", map_location="cpu", weights_only=False
            )
            model = build_model(num_labels)
            model.load_state_dict(state["state_dict"])
            model.to(device)
            best_f1 = float(state["val_macro_f1"])
        else:
            train_dataset = ImageItems(
                train_items, image_getter, label_getter, train=True
            )
            val_dataset = ImageItems(
                val_items, image_getter, label_getter, train=False
            )
            model, _, best_f1 = train_one(
                seed=seed,
                num_labels=num_labels,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_labels=train_labels,
                device=device,
                epochs=epochs,
                output_dir=seed_dir,
            )
        result = evaluate_seed(
            model=model,
            seed=seed,
            train_items=train_items,
            val_items=val_items,
            test_items=test_items,
            image_getter=image_getter,
            label_getter=label_getter,
            false_critical=false_critical,
            device=device,
            output_dir=seed_dir,
        )
        result["best_val_macro_f1"] = best_f1
        result["runtime_seconds"] = time.time() - started
        (seed_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        seed_results[str(seed)] = result
        print(
            "SEED_COMPLETE",
            name,
            seed,
            json.dumps(result["s8"], sort_keys=True),
            flush=True,
        )
    return {"seeds": seed_results, "aggregate": aggregate(seed_results)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=["aider", "hurricane"], required=True
    )
    parser.add_argument("--aider-epochs", type=int, default=6)
    parser.add_argument("--hurricane-epochs", type=int, default=5)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/derived/greatness_strengthening",
    )
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "primary_multiseed.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary["protocol"] = {
        "seeds": list(SEEDS),
        "scales": list(SCALES),
        "gamma": GAMMA,
        "knn_k": 10,
        "device": str(device),
        "pretrained_model": str(PRETRAINED),
    }
    for name in args.datasets:
        epochs = args.aider_epochs if name == "aider" else args.hurricane_epochs
        summary[name] = run_dataset(
            name, epochs, device, args.output, eval_only=args.eval_only
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("PRIMARY_MULTISEED_COMPLETE", summary_path, flush=True)


if __name__ == "__main__":
    main()
