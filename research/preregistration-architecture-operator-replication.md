# Preregistration: architecture and degradation-operator replication

Status: **FROZEN before MobileNet training or evaluation**  
Date: 2026-07-25

## Motivation

Independent greatness review found that the central oracle--received protocol
gap used one ResNet-18 architecture and one bicubic degradation family. This
prospective replication tests whether the sign persists with a second
architecture and four fixed image-resampling operators.

## Data and split

- Use the exact-hash-clean AIDER split already registered:
  4,499 training, 961 validation, 969 test images.
- No split changes, test filtering, or test-dependent model selection.
- Dataset root is provided by `AIDER_ROOT`; all released paths remain relative.

## Model and training

- Architecture: torchvision MobileNetV3-Small with ImageNet-1K initialisation.
- Independent seeds: `{101,202,303}`.
- Six epochs; AdamW learning rate `3e-4`, weight decay `1e-4`, batch 64.
- Class-weighted cross-entropy with weight `N/(C n_c)`.
- Best source-validation macro-F1 checkpoint selected within each seed.

## Fixed operators and scores

Operators: bicubic, bilinear, nearest-neighbour, and box down/up sampling.

For each operator and test image, compute predictions at effective scales
`{1,2,4,8,16,32,64}`. Errors are those of the `s=8` prediction.

- **Reference-dependent score:** the original `s=1`-anchored mean JS score over
  `{2,4,8}`.
- **Received-image score:** `s=8`-anchored mean JS over `{16,32,64}`.
- **Anchor-matched fine-reference sensitivity:** `s=8`-anchored mean JS over
  `{1,2,4}`.
- Confidence baseline: one minus maximum softmax probability at `s=8`.

Report per-example arrays, AUROC, AUPRC, FPR95, seed means/sample SD, and
2,000-replicate paired bootstrap intervals.

## Criteria

- **M1 architecture replication:** under bicubic degradation, the
  anchor-matched fine-reference minus received-image AUROC gap is positive for
  all three MobileNet seeds and its paired interval excludes zero for each.
- **M2 operator replication:** for each of the four operators, the mean
  anchor-matched gap across MobileNet seeds is positive and at least two of
  three seed intervals exclude zero.
- **M3 cross-architecture sign:** ResNet-18 and MobileNetV3-Small have positive
  mean anchor-matched gaps for all four operators. For each ResNet
  corpus/operator combination, the mean is positive and at least two of three
  seed intervals exclude zero.

Failure is retained and blocks claims of architecture/operator generality. No
criterion is changed after test evaluation.

## Pre-evaluation implementation addendum

The torchvision ImageNet weight endpoint returned HTTP 503 on two runner
attempts and ten direct retries before any model was trained or evaluated.
The architecture is therefore implemented with
`timm==1.0.28` model `mobilenetv3_small_100`, using its public ImageNet-1K
weights. Optimisation, seeds, split, epochs, operators, endpoints and M1--M3
criteria are unchanged. Both failed download attempts are retained.
