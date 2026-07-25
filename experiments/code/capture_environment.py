#!/usr/bin/env python3
"""Capture exact software, model, and dataset revisions."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import datasets
import huggingface_hub
import matplotlib
import numpy
import PIL
import sklearn
import timm
import torch
import torchvision
import transformers
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "numpy": numpy.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__,
        "matplotlib": matplotlib.__version__,
        "device": "Apple M4 Max MPS",
        "resnet18_model": "microsoft/resnet-18",
        "resnet18_revision": "65a5785d9156231087c481e0c7dd33a5ff6f7e3e",
        "mobilenet_model": "timm/mobilenetv3_small_100",
        "mobilenet_weights": "ImageNet-1K pretrained",
        "hurricane_dataset_revision": HfApi()
        .dataset_info(
            "jonathan-roberts1/Satellite-Images-of-Hurricane-Damage"
        )
        .sha,
        "crasar_dataset_revision": (
            "47cf4ab3a94d42978975f7d23338a996125ac0e9"
        ),
    }
    output = ROOT / "experiments/derived/environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print("ENVIRONMENT_CAPTURE_COMPLETE", json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
