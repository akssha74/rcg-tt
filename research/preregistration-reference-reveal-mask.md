# Preregistration: fine-reference reveal/mask identification experiment

Status: **FROZEN before permutation evaluation**  
Date: 2026-07-25

## Motivation

Independent scientific review found that the oracle--received protocol gap
changes reference availability and degradation direction together. The
architecture/operator replication establishes robustness but does not isolate
the image-specific information supplied by the corresponding finer views.

## Fixed design

Use the exact-hash-clean AIDER and Hurricane splits, ResNet-18 seeds
`{101,202,303}`, and the same four operators: bicubic, bilinear,
nearest-neighbour, and box.

For each test image \(i\):

1. The error label is fixed from the \(s=8\) prediction.
2. The anchor is fixed at \(p_{8,i}\).
3. The aligned fine-reference score is
   \[
   D_{\mathrm{aligned},i} =
   \frac{1}{3}\sum_{s\in\{1,2,4\}}
   \operatorname{JS}(p_{8,i},p_{s,i}).
   \]
4. For each of 100 fixed random permutations \(\pi_b\), the masked score is
   \[
   D_{\mathrm{masked},i}^{(b)} =
   \frac{1}{3}\sum_{s\in\{1,2,4\}}
   \operatorname{JS}(p_{8,i},p_{s,\pi_b(i)}).
   \]

The masked score retains the same anchor, fine-scale candidate set, score
functional, comparison count, marginal fine-prediction distributions and error
set. It removes only correspondence between a coarse image and its own finer
views. Permutations are global, generated with seed `260725`, and shared across
the three fine scales within a replicate.

## Endpoints

Per corpus, seed, and operator report:

- aligned AUROC;
- mean, SD, minimum and maximum masked AUROC across 100 permutations;
- aligned-minus-masked-mean AUROC;
- empirical one-sided permutation probability
  \((1 + \#\{A_{\mathrm{masked}}\ge A_{\mathrm{aligned}}\})/101\);
- raw aligned/masked scores and error labels.

## Criterion M4

For all 24 corpus/seed/operator combinations:

- aligned AUROC exceeds mean masked AUROC;
- empirical probability is at most `0.05`.

If any combination fails, the paper must retain the result and cannot claim
that corresponding fine-reference information consistently drives the
diagnostic advantage. This experiment does not make the score deployable.
