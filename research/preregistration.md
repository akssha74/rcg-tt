# Preregistration: information-matched resolution reliability audit

Status: **FROZEN before sibling-study outcome analysis**  
Date: 2026-07-24

## Imported failed branch

The sibling imports the prior RCG-GSD branch only as versioned evidence. The
prior branch observed test outcomes and therefore cannot supply a new-method
confirmatory claim. Its preregistered received-image W8/W9 failure is a primary
negative result. New corrective runs are executed under
`studies/disaster-rcg-gsd/research/preregistration-leakage-free-replication.md`
and are imported only with source hashes, successful run records, and explicit
supersession links.

## Data and split integrity

- AIDER: preserve the frozen split except that every member of any
  conflicting-label exact SHA-256 duplicate group is excluded. Require zero
  remaining cross-split hashes.
- Hurricane Damage: compute encoded-image SHA-256 for the fixed seeded
  70/10/20 index split. Exclude every member of any cross-split duplicate group
  and retrain if exclusions are nonzero.
- CRASAR-U-DROIDs: use guarded 2048-pixel source-image blocks for model
  training, excluding 512-pixel crops within 256 pixels of a block boundary.
  Require zero exhaustive cross-split rectangle intersections.
- Measured-GSD evaluation uses four fully held-out paired sites from Hurricane
  Ian and Hurricane Michael, with identical eligible labels across UAS and
  post-event satellite products.

## Models

ImageNet-initialised ResNet-18 seeds `{101,202,303}` are primary. AIDER uses six
epochs. Leakage-free CRASAR uses 12 epochs and training sites fixed in the
corrective preregistration. Hurricane uses five epochs unless its hash audit
requires retraining, in which case the architecture and optimisation remain
unchanged.

## Information sets and scores

At a received operating scale `s`, all deployable scores use only that received
image, source training data, and source validation data:

1. one minus maximum softmax probability;
2. negative maximum logit;
3. energy score;
4. EO deep nearest-neighbour feature distance, with
   `k in {1,5,10,20,50}` selected by source-validation error AUROC;
5. virtual-logit matching (ViM), fitted only to source training features and
   classifier weights, with principal dimension selected from
   `{64,128,256,384}` by source-validation AUROC;
6. received-image consistency: mean Jensen--Shannon divergence between the
   received prediction and predictions after relative degradations `{2,4,8}`.

The privileged diagnostic additionally compares a synthetically degraded
received image at `s=8` with predictions from original scales `{1,2,4,8}`. It
is labelled non-deployable and is never ranked as an operational method.

No score may use target labels, target calibration, an unavailable finer image,
or another method's larger information set. KNN `k`, ViM dimension, temperature,
and any score orientation are frozen from source training/validation only.

## Endpoints and uncertainty

For each corpus and seed report accuracy, error prevalence, AUROC, AUPRC, and
FPR95 for every score. Report paired within-seed bootstrap intervals (10,000
replicates; seed 260724 plus model seed) for:

- privileged consistency minus received-image consistency;
- each deployable score minus confidence;
- ViM minus EO-kNN.

For threshold transfer, calibrate each score on source validation at target
coverages `{0.5,0.7,0.9}` and report unchanged-threshold test coverage,
absolute coverage error, selective risk, and false-critical rate.

Measured-GSD inference follows the corrective preregistration and uses paired
spatial-cluster bootstrap by site and 2048-pixel source block.

## Fixed audit criteria

- **A1 privileged inflation:** privileged-minus-received consistency mean AUROC
  is positive on both AIDER and Hurricane, both paired bootstrap 95% intervals
  exclude zero, and all six seed differences are positive.
- **A2 matched baseline completeness:** confidence, max-logit, energy, EO-5NN,
  ViM, and received consistency are present for every primary seed with
  identical received-image access.
- **A3 leakage closure:** zero AIDER/Hurricane cross-split hashes and zero
  CRASAR cross-split crop intersections after the fixed exclusions/guards.
- **A4 measured-GSD validity:** at least four paired sites and two events;
  pooled UAS accuracy exceeds the majority baseline for all seeds; mean
  balanced accuracy is at least 0.60; site and spatial-cluster intervals are
  reported.
- **A5 negative-result integrity:** failed W6, W8, W9, measured threshold
  transfer, contaminated estimates, and all corrective failures remain
  discoverable and are not relabelled as confirmatory successes.

Failure of A1 kills this sibling claim. Failure of A2 or A3 invalidates the
audit. Failure of A4 blocks operational measured-GSD conclusions and triggers
one preregistered data/model correction; a second failure stops the branch.
# Preregistration

Status: DRAFT — freeze commit/hash before confirmatory evaluation.

## Hypothesis and mechanism

## Dataset versions and frozen splits

## Methods and baselines

## Primary and secondary outcomes

## Seeds, sample size, and stopping rule

## Statistical analysis and uncertainty

## Hyperparameter selection and equal-budget policy

## Ablations and robustness checks

## Inclusion, exclusion, and failed-run handling

## Compute budget

## Deviations log
