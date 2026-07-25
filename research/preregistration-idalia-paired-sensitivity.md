# Preregistration: third-event paired aerial sensitivity

Status: **FROZEN before model evaluation**  
Date: 2026-07-25

## Motivation

Independent greatness review found that measured-GSD evidence covered only
Hurricane Ian and Hurricane Michael. CRASAR-U-DROIDs also provides held-out,
post-event Hurricane Idalia UAS and crewed-aircraft products for Steinhatchee
River with measured GSD and shared building identifiers.

## Fixed data

- UAS: `20230830-SteinhatcheeRiver.geo.tif`, GSD 0.127000 m/px.
- Crewed: `20230830-SteinhatcheeRiver.geo.tif_20230831a_RGB.geo.tif`,
  GSD 0.150161 m/px.
- Dataset revision:
  `47cf4ab3a94d42978975f7d23338a996125ac0e9`.
- Retain common building IDs with identical eligible labels, and report all
  selection counts.
- Extract 512-by-512 crops in each product; record physical footprints and
  joint UAS/crewed overlap components.

## Models and scores

Use the frozen intersection-controlled CRASAR ResNet-18 checkpoints for seeds
`{101,202,303}`. Compute confidence and received-image consistency from
relative scales `{1,2,4,8}` independently within each received product. No
fine reference, target fitting, or threshold recalibration is allowed.

## Endpoints

- accuracy, balanced accuracy, macro-F1 and majority accuracy;
- error AUROC/AUPRC for confidence and received consistency;
- paired UAS-to-crewed accuracy change;
- received-consistency minus confidence AUROC;
- per-example predictions/scores and exact transitions;
- joint-overlap-cluster bootstrap intervals, with degeneracy reported.

## Criteria

- **E1 classifier validity:** mean UAS balanced accuracy is at least 0.55 and
  all three seed UAS accuracies are reported relative to majority accuracy.
- **E2 third-event completeness:** all metrics, raw arrays, selection counts,
  physical footprints and cluster counts are produced for all three seeds.

No positive ranking or accuracy-direction criterion is imposed. The experiment
tests generality and must retain null or reversed results.
