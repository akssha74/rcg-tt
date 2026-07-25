# Preregistration: leakage-free measured-GSD replication

Status: **FROZEN before data preparation, training, or evaluation**  
Date: 2026-07-24

## Motivation

The round-5 review identified three defects in the measured-GSD evidence:
cross-split overlap between 512-pixel crops, a classifier below the majority
accuracy baseline, and inference based on two sites from one event without
paired spatial uncertainty. This corrective replication replaces the affected
splits and checkpoints rather than reusing them.

## AIDER duplicate correction

- Compute SHA-256 over the encoded bytes of every image in the frozen AIDER
  split.
- If one hash occurs with conflicting class labels, exclude every member of
  that hash group. Preserve all other train/validation/test assignments.
- Assert that no remaining hash appears in more than one split.
- Retrain ResNet-18 seeds `{101,202,303}` for six epochs with the previously
  fixed optimiser, class weighting, and model-selection metric.
- Treat all earlier AIDER checkpoints and endpoint values as superseded.

## CRASAR training data

Use CRASAR-U-DROIDs snapshot
`47cf4ab3a94d42978975f7d23338a996125ac0e9`. The fixed UAS training sites are:

- `090403-Lancaster-Canyon-Gate.geo.tif`;
- `1001-Summerlin-San-Carlos.geo.tif`;
- `1001-Palmeto-Palms.geo.tif`;
- `1001-Kennedy-Green-Mobile-Homes.geo.tif`;
- `1001-Ft-Myers-Beach-DIRT.geo.tif`.

Extract one 512-by-512 RGB crop per eligible labelled building. Assign
2048-by-2048 source-image blocks deterministically by SHA-256 to 60% training,
20% validation, and 20% internal audit. Exclude crops whose centroid is within
256 pixels of any block boundary. This guard implies that crops from
differently assigned adjacent blocks cannot overlap. Verify the implication by
an exhaustive rectangle-intersection audit and require zero cross-split
intersections before training.

Train ImageNet-initialised ResNet-18 seeds `{101,202,303}` for 12 epochs using
class-weighted cross-entropy, the fixed augmentation pipeline, and best
validation macro-F1 checkpoint selection. Do not use evaluation-site images,
labels, or outcomes for training or model selection.

## Held-out paired measured-GSD evaluation

The fixed evaluation sites are:

- Hurricane Ian: `1001-Harlem-Heights.geo.tif` and
  `1001-McGregor-College-Pkwy-South.1.geo.tif`, paired with the post-event
  MAXAR `10300100DB06A700` products;
- Hurricane Michael: `10132018-MexicoBeach.geo.tif` and
  `10142018-MexicoBeach.geo.tif`, paired with the post-event MAXAR
  `104001004384D900` products.

Require identical eligible damage labels for each UAS/satellite building pair.
The two Mexico Beach sites are untouched by previous model training and extend
evaluation to a second event, platform, and satellite GSD. Report each site
separately and pool only after retaining site identifiers.

For every seed and modality report accuracy, balanced accuracy, macro-F1,
error AUROC/AUPRC for confidence and received-image RCG, and the majority-class
accuracy baseline. RCG uses only relative degradations `{1,2,4,8}` constructed
from the received crop.

Calibrate confidence and RCG thresholds only on the pooled guarded validation
set at target coverages `{0.5,0.7,0.9}`. Transfer thresholds unchanged to each
held-out UAS and satellite site. Report realised coverage, coverage error,
selective risk, and false-critical rate.

Use a paired spatial-cluster bootstrap with sites as top-level clusters and
2048-by-2048 UAS source-image blocks as within-site resampling units (10,000
replicates, seed 260724) for accuracy change, AUROC difference, coverage
difference, and selective-risk difference. Keep every building pair in a
sampled block together. Also report exact per-building paired prediction
transitions.

## Fixed criteria

- **W10 leakage:** zero cross-split encoded-byte duplicates on AIDER and zero
  cross-split 512-pixel rectangle intersections on CRASAR.
- **W11 classifier validity:** pooled held-out UAS accuracy exceeds the pooled
  majority-class baseline for all three seeds, and mean balanced accuracy is at
  least 0.60.
- **W12 deployable ranking:** mean received-image RCG error AUROC exceeds
  confidence on held-out satellite imagery, the paired cluster-bootstrap 95%
  interval for the pooled difference excludes zero, and at least three of four
  sites have a positive mean difference.
- **W13 threshold transfer:** at target coverage 0.70, mean satellite absolute
  coverage error is no greater for RCG than confidence and mean RCG selective
  risk is no higher.

Failure of W10 invalidates the run. Failure of W11 prevents operational use of
the reliability results. Failure of W12 or W13 must be retained and triggers a
method pivot; it cannot be repaired by claim wording.
