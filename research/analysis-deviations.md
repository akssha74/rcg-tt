# Analysis deviations and post-hoc sensitivities

Date recorded: 2026-07-25

## W13 adjudication correction

The frozen corrective preregistration defines W13 on **mean** satellite
absolute coverage error and **mean** selective risk across seeds. The first
implementation instead required both inequalities to hold for every seed and
stored that stricter outcome as `w13_threshold_transfer`.

The mean-based preregistered rule passes:

- mean absolute coverage error: received consistency 0.1056, confidence 0.2006;
- mean selective risk: received consistency 0.4563, confidence 0.5479.

The all-seed robustness sensitivity fails because seed 202 has selective risk
0.5488 for received consistency and 0.5305 for confidence. The manuscript must
report W13 as passed and label the all-seed outcome as post hoc.

## Independent-review sensitivities

The following analyses were added after independent review and are not promoted
to preregistered confirmatory endpoints:

1. an anchor-matched fine-reference score using the received `s=8` prediction
   as anchor and the same three-comparison count as received consistency;
2. per-example score and prediction archives;
3. paired-site selection counts and exact prediction transitions;
4. site-specific intervals and joint UAS/satellite overlap-cluster bootstrap;
5. archive portability, environment locking, and clean end-to-end reproduction.

The original summaries, failed branches, and preregistered outcomes remain
retained.

## MobileNet weight source

The prospective architecture replication initially specified torchvision
MobileNetV3-Small ImageNet weights. The PyTorch model endpoint returned HTTP
503 on two runner attempts and ten direct retries before training began. The
frozen pre-evaluation addendum switched to
`timm==1.0.28` `mobilenetv3_small_100` ImageNet-1K weights without changing
seeds, optimisation, splits, operators, endpoints or M1--M3. Both failed runs
are retained in the run ledger.
