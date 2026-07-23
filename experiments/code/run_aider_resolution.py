#!/usr/bin/env python3
"""AIDER resolution-ladder experiments: train, evaluate, RCG vs confidence."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Split files in this public release store dataset-relative image paths
# (e.g. "collapsed_building/collapsed_building_image0024.jpg"). Set the
# environment variable AIDER_ROOT to your local AIDER root so relative paths
# resolve; absolute paths (from a locally regenerated split) are used as-is.
AIDER_ROOT = os.environ.get("AIDER_ROOT", "")


def resolve_image_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute() or not AIDER_ROOT:
        return path
    return str(Path(AIDER_ROOT) / p)

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Order matches the local AIDER MobileNet checkpoint class list.
CLASS_NAMES = [
    "collapsed_building",
    "fire",
    "flooded_areas",
    "traffic_incident",
    "normal",
]
DISASTER = {0, 1, 2, 3}
NORMAL = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_aider(root: Path) -> Dict[str, List[Path]]:
    out = {}
    for name in CLASS_NAMES:
        d = root / name
        files = sorted(
            [p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        out[name] = files
    return out


def make_splits(root: Path, out_path: Path, seed: int = 42) -> dict:
    rng = random.Random(seed)
    by = list_aider(root)
    splits = {"train": [], "val": [], "test": [], "seed": seed, "class_names": CLASS_NAMES}
    for ci, name in enumerate(CLASS_NAMES):
        files = by[name][:]
        rng.shuffle(files)
        n = len(files)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        parts = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val :],
        }
        for split, fl in parts.items():
            for p in fl:
                splits[split].append({"path": str(p), "label": ci, "class": name})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=2))
    return splits


class AiderDataset(Dataset):
    def __init__(self, items, scale: int = 1, train: bool = False):
        self.items = items
        self.scale = scale
        if train:
            self.tf = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.tf = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(resolve_image_path(it["path"])).convert("RGB")
        if self.scale > 1:
            w, h = img.size
            nw, nh = max(1, w // self.scale), max(1, h // self.scale)
            img = img.resize((nw, nh), Image.BICUBIC).resize((w, h), Image.BICUBIC)
        x = self.tf(img)
        y = it["label"]
        return x, y, it["path"]


def build_model(num_classes=5):
    # Local network to download.pytorch.org is unavailable in this environment;
    # train MobileNetV3-Small from scratch (documented in prereg deviations).
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs_all, ys, paths = [], [], []
    for x, y, p in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs_all.append(probs)
        ys.append(y.numpy())
        paths.extend(p)
    return np.concatenate(probs_all), np.concatenate(ys), paths


def train_one(splits, out_dir: Path, seed: int, device, epochs=12, multi_res=False):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_items = splits["train"]
    val_items = splits["val"]
    train_ds = AiderDataset(train_items, scale=1, train=True)
    val_ds = AiderDataset(val_items, scale=1, train=False)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_f1, best_path = -1.0, out_dir / f"model_seed{seed}.pt"
    history = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y, _ in train_loader:
            if multi_res:
                # resolution curriculum: random scale in {1,2,4}
                s = random.choice([1, 2, 4])
                if s > 1:
                    # approximate by pooling then upsample in tensor space
                    x = torch.nn.functional.interpolate(
                        torch.nn.functional.interpolate(
                            x, scale_factor=1 / s, mode="bilinear", align_corners=False, recompute_scale_factor=True
                        ),
                        size=x.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        probs, ys, _ = predict(model, val_loader, device)
        pred = probs.argmax(1)
        # macro-F1
        f1s = []
        for c in range(5):
            tp = np.sum((pred == c) & (ys == c))
            fp = np.sum((pred == c) & (ys != c))
            fn = np.sum((pred != c) & (ys == c))
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1s.append(2 * prec * rec / (prec + rec + 1e-9))
        macro = float(np.mean(f1s))
        history.append({"epoch": ep, "train_loss": total / len(train_items), "val_macro_f1": macro})
        if macro > best_f1:
            best_f1 = macro
            torch.save({"model": model.state_dict(), "seed": seed, "val_macro_f1": macro}, best_path)
    return best_path, history, best_f1


def js_divergence(p, q, eps=1e-9):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.sum(a * np.log(a / b), axis=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def evaluate_ladder(model, items, device, scales=(1, 2, 4, 8)):
    results = {}
    for s in scales:
        ds = AiderDataset(items, scale=s, train=False)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        probs, ys, paths = predict(model, loader, device)
        pred = probs.argmax(1)
        conf = probs.max(1)
        acc = float((pred == ys).mean())
        # false critical: pred disaster, true normal
        fc = float(np.mean((ys == NORMAL) & np.isin(pred, list(DISASTER))))
        # dangerous miss: true disaster, pred normal
        dm = float(np.mean((ys != NORMAL) & (pred == NORMAL)))
        err_conf = float(conf[pred != ys].mean()) if np.any(pred != ys) else float("nan")
        results[str(s)] = {
            "accuracy": acc,
            "macro_acc": acc,
            "mean_confidence": float(conf.mean()),
            "mean_confidence_on_errors": err_conf,
            "false_critical_rate": fc,
            "dangerous_miss_rate": dm,
            "probs": probs,
            "ys": ys,
            "pred": pred,
            "conf": conf,
            "paths": paths,
        }
    return results


def disagreement_scores(ladder):
    base = ladder["1"]
    ys = base["ys"]
    d_hard = np.zeros(len(ys))
    d_js = np.zeros(len(ys))
    n = 0
    for s, res in ladder.items():
        if s == "1":
            continue
        n += 1
        d_hard += (res["pred"] != base["pred"]).astype(float)
        d_js += js_divergence(base["probs"], res["probs"])
    d_hard /= max(n, 1)
    d_js /= max(n, 1)
    errors = (base["pred"] != ys).astype(int)
    return d_hard, d_js, errors, base


def auroc_safe(y, score):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def selective_curves(keep_score, is_false_critical, n_grid=21):
    # higher keep_score => more likely auto-decide; for confidence use conf; for RCG use -disagreement
    qs = np.linspace(0, 1, n_grid)
    thr = np.quantile(keep_score, qs)
    out = []
    for t in thr:
        auto = keep_score >= t
        cov = float(auto.mean())
        if auto.sum() == 0:
            fcr = 0.0
        else:
            fcr = float(is_false_critical[auto].mean())
        out.append({"threshold": float(t), "coverage": cov, "false_critical_rate": fcr})
    return out


def bootstrap_auroc_diff(y, s1, s2, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    if not diffs:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    diffs = np.array(diffs)
    return {
        "mean": float(diffs.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }


def match_coverage_fcr(curve_a, curve_b, target=None):
    # compare FCR at closest coverages; also interpolate relative reduction at max mutual coverage grid
    rows = []
    for a in curve_a:
        # find b with closest coverage
        b = min(curve_b, key=lambda x: abs(x["coverage"] - a["coverage"]))
        if b["false_critical_rate"] <= 1e-12 and a["false_critical_rate"] <= 1e-12:
            rel = 0.0
        elif b["false_critical_rate"] <= 1e-12:
            rel = float("nan")
        else:
            rel = (b["false_critical_rate"] - a["false_critical_rate"]) / b["false_critical_rate"]
        rows.append(
            {
                "coverage_a": a["coverage"],
                "coverage_b": b["coverage"],
                "fcr_a": a["false_critical_rate"],
                "fcr_b": b["false_critical_rate"],
                "relative_reduction_a_vs_b": rel,
            }
        )
    # summarize at coverage closest to 0.7
    mid = min(rows, key=lambda r: abs(r["coverage_a"] - 0.7))
    return {"pairs": rows, "at_coverage_~0.7": mid}


def strip_arrays(ladder):
    out = {}
    for k, v in ladder.items():
        out[k] = {kk: vv for kk, vv in v.items() if kk not in {"probs", "ys", "pred", "conf", "paths"}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aider-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--multi-res-train", action="store_true")
    args = ap.parse_args()

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    split_path = work / "configs" / "aider_splits.json"
    if not split_path.exists():
        # configs may live under experiments/
        split_path = work.parent / "configs" / "aider_splits.json" if False else split_path
    # Prefer study experiments layout
    cfg_dir = work / "configs" if (work / "configs").exists() or True else work
    cfg_dir = work if work.name == "configs" else (work / "configs")
    # simplify: write next to derived
    split_path = Path(args.work) / "aider_splits.json"
    splits = make_splits(args.aider_root, split_path, seed=42)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    summary = {"device": str(device), "seeds": {}, "aggregate": {}}

    seed_rows = []
    for seed in args.seeds:
        t0 = time.time()
        model_dir = Path(args.work) / f"seed_{seed}"
        ckpt, hist, best_f1 = train_one(
            splits, model_dir, seed, device, epochs=args.epochs, multi_res=args.multi_res_train
        )
        model = build_model()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        model.to(device)
        ladder = evaluate_ladder(model, splits["test"], device)
        d_hard, d_js, errors, base = disagreement_scores(ladder)
        # For RCG keep score = 1 - d_hard; confidence keep = conf
        is_fc = (base["ys"] == NORMAL) & np.isin(base["pred"], list(DISASTER))
        # error detection scores: higher => more likely error
        auroc_conf = auroc_safe(errors, 1.0 - base["conf"])  # low conf => error? actually use -conf
        # standard: score high means predicted error; use (1-conf) and d_hard
        auroc_conf = auroc_safe(errors, 1.0 - base["conf"])
        auroc_d = auroc_safe(errors, d_hard)
        auroc_js = auroc_safe(errors, d_js)
        diff = bootstrap_auroc_diff(errors, d_hard, 1.0 - base["conf"], seed=seed)
        curve_conf = selective_curves(base["conf"], is_fc)
        curve_rcg = selective_curves(1.0 - d_hard, is_fc)
        matched = match_coverage_fcr(curve_rcg, curve_conf)
        row = {
            "seed": seed,
            "val_macro_f1": best_f1,
            "train_seconds": time.time() - t0,
            "ladder": strip_arrays(ladder),
            "auroc_error_given_one_minus_conf": auroc_conf,
            "auroc_error_given_disagreement": auroc_d,
            "auroc_error_given_js": auroc_js,
            "auroc_lift_disagreement_minus_one_minus_conf": auroc_d - auroc_conf,
            "bootstrap_auroc_lift": diff,
            "selective_confidence": curve_conf,
            "selective_rcg": curve_rcg,
            "matched_coverage": matched["at_coverage_~0.7"],
            "n_test": int(len(base["ys"])),
        }
        # persist arrays for analysis
        np.savez_compressed(
            model_dir / "test_arrays.npz",
            d_hard=d_hard,
            d_js=d_js,
            errors=errors,
            conf=base["conf"],
            ys=base["ys"],
            pred=base["pred"],
            is_fc=is_fc.astype(np.uint8),
        )
        (model_dir / "metrics.json").write_text(json.dumps(row, indent=2))
        (model_dir / "history.json").write_text(json.dumps(hist, indent=2))
        summary["seeds"][str(seed)] = row
        seed_rows.append(row)
        print(json.dumps({"seed": seed, "lift": row["auroc_lift_disagreement_minus_one_minus_conf"], "matched": row["matched_coverage"]}, indent=2))

    lifts = [r["auroc_lift_disagreement_minus_one_minus_conf"] for r in seed_rows]
    rels = [r["matched_coverage"]["relative_reduction_a_vs_b"] for r in seed_rows]
    summary["aggregate"] = {
        "auroc_lift_mean": float(np.mean(lifts)),
        "auroc_lift_std": float(np.std(lifts)),
        "relative_fcr_reduction_at_cov0.7_mean": float(np.nanmean(rels)),
        "relative_fcr_reduction_at_cov0.7_std": float(np.nanstd(rels)),
        "ladder_acc_seed42": seed_rows[0]["ladder"] if seed_rows else {},
    }
    out = Path(args.work) / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print("WROTE", out)


if __name__ == "__main__":
    main()
