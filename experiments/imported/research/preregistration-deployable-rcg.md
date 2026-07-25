# Preregistration: deployable received-image RCG

Status: **FROZEN before evaluation**  
Date: 2026-07-24

## Motivation

The independent default-reject reviews found that the prior synthetic s=8 score
used the unavailable original s=1 view. This experiment removes that privileged
information.

## Fixed protocol

- Start from the image available to the deployed system at each operating scale
  $s_0$.
- Construct only additional degradations of that received image at relative
  factors `{1,2,4,8}`; at original scale 8 this corresponds to effective scales
  `{8,16,32,64}`.
- The pairwise-native JS reference is the received image itself, never an
  unavailable finer view.
- Errors, confidence, EO 5-NN, and RCG are evaluated on identical received
  images and identical model checkpoints.
- EO nearest-neighbour baseline: normalized penultimate features, global native
  training bank, fixed `k=5` from Dimitric et al.; no target/test fitting.
- Use the existing AIDER and Hurricane ResNet seeds `{101,202,303}`.
- Report seed values, mean/sample SD, paired within-seed bootstrap intervals,
  AUPRC, and FPR at 95% error recall.

## Threshold transfer

- Calibrate confidence, EO-5NN and deployable-RCG thresholds on native
  validation at target coverage 0.7.
- Transfer unchanged to received s=8 test images.
- Report coverage and FCR for all gates and all seeds.

## Criterion W8

- Mean deployable-RCG AUROC lift over confidence and EO-5NN is positive on both
  corpora; and
- at least two of three seeds per corpus have positive lift over each baseline.

## Criterion W9

- Deployable-RCG transferred coverage exceeds confidence coverage on both
  corpora without higher mean FCR.

If W8 or W9 fails, preserve the result and redesign or narrow the operational
claim; do not reuse the privileged-view score as deployment evidence.
