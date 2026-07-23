#!/usr/bin/env python3
"""TTA baseline and high-frequency energy mechanism probe on a trained AIDER seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import run_aider_resolution as base


class TTADataset(Dataset):
    def __init__(self, items):
        self.items = items
        self.base = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["path"]).convert("RGB")
        xs = [self.base(img), self.base(img.transpose(Image.FLIP_LEFT_RIGHT))]
        # two mild crops via resize variants
        xs.append(self.base(img.resize((int(img.size[0] * 0.9), int(img.size[1] * 0.9)))))
        y = it["label"]
        return torch.stack(xs), y


def hf_energy(path: str) -> float:
    img = Image.open(path).convert("L").resize((256, 256))
    arr = np.asarray(img, dtype=np.float32)
    f = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.abs(f)
    yy, xx = np.mgrid[-128:128, -128:128]
    r = np.sqrt(xx**2 + yy**2)
    mask = r > 64
    return float(mag[mask].mean() / (mag.mean() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    splits = json.loads((args.work / "aider_splits.json").read_text())
    ckpt = args.work / f"seed_{args.seed}" / f"model_seed{args.seed}.pt"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = base.build_model()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    ds = TTADataset(splits["test"])
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    preds_modes = []
    ys_all = []
    agree = []
    with torch.no_grad():
        for xs, y in loader:
            # xs: B,T,C,H,W
            b, t = xs.shape[:2]
            logits = model(xs.view(b * t, *xs.shape[2:]).to(device))
            probs = torch.softmax(logits, dim=-1).view(b, t, -1).cpu().numpy()
            pred_t = probs.argmax(-1)
            mode = np.array([np.bincount(pred_t[i], minlength=5).argmax() for i in range(b)])
            # disagreement among TTA views
            d = np.mean(pred_t != mode[:, None], axis=1)
            preds_modes.append(mode)
            ys_all.append(y.numpy())
            agree.append(d)
    pred = np.concatenate(preds_modes)
    ys = np.concatenate(ys_all)
    d_tta = np.concatenate(agree)
    errors = (pred != ys).astype(int)
    auroc_tta = float(roc_auc_score(errors, d_tta)) if len(np.unique(errors)) > 1 else float("nan")

    # HF energy on native images
    energies = np.array([hf_energy(it["path"]) for it in splits["test"]])
    # correlate low HF with errors of native model
    native_ds = base.AiderDataset(splits["test"], scale=1, train=False)
    native_loader = DataLoader(native_ds, batch_size=64, shuffle=False, num_workers=0)
    probs, ys2, paths = base.predict(model, native_loader, device)
    pred2 = probs.argmax(1)
    errors2 = (pred2 != ys2).astype(int)
    # low HF => higher error risk => score = -energy
    auroc_hf = float(roc_auc_score(errors2, -energies)) if len(np.unique(errors2)) > 1 else float("nan")

    # also under scale 4
    ladder4 = base.AiderDataset(splits["test"], scale=4, train=False)
    loader4 = DataLoader(ladder4, batch_size=64, shuffle=False, num_workers=0)
    probs4, ys4, _ = base.predict(model, loader4, device)
    err4 = (probs4.argmax(1) != ys4).astype(int)
    auroc_hf4 = float(roc_auc_score(err4, -energies)) if len(np.unique(err4)) > 1 else float("nan")

    out = {
        "seed": args.seed,
        "tta_accuracy": float((pred == ys).mean()),
        "auroc_error_given_tta_disagreement": auroc_tta,
        "auroc_error_given_neg_hf_energy_native": auroc_hf,
        "auroc_error_given_neg_hf_energy_scale4": auroc_hf4,
        "mean_hf_energy": float(energies.mean()),
        "mean_hf_energy_on_errors_native": float(energies[errors2 == 1].mean()) if errors2.any() else None,
        "mean_hf_energy_on_correct_native": float(energies[errors2 == 0].mean()) if (errors2 == 0).any() else None,
    }
    path = args.work / f"seed_{args.seed}" / "tta_hf_probe.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
