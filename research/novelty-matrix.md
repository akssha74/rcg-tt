# Claim-level novelty matrix

## Predictive-distribution disagreement

- **Closest work:** Schirmer, Zhang and Nalisnick (2023) compare JS, KL and
  Hellinger disagreement across multiple models for performance forecasting
  under distribution shift.
- **Also relevant:** AugMix uses JS consistency as a training loss across
  augmentations.
- **Not novel here:** Jensen--Shannon divergence, disagreement as uncertainty,
  or consistency across transformed predictions.
- **Defensible delta:** quantify the oracle--received protocol gap when one
  transformation is an unavailable finer reference; compare the same score
  after information matching and with a post-hoc matched-anchor sensitivity.

## Earth-observation OOD baselines

- **Closest work:** DPN-RS studies sensor, location and class shift with
  auxiliary OOD training; Li et al. compare ten EO OOD methods and identify ViM
  and KNN as leading no-retraining methods. TARDIS uses known/wild target pools;
  EarthShift benchmarks real EO shifts including resolution.
- **Not novel here:** ViM, KNN, confidence, energy, or generic OOD scoring.
- **Defensible delta:** compare these methods for *prediction-error detection
  under resolution shift* with an identical received-image information set,
  rather than presenting consistency with a hidden reference as a fair OOD
  competitor.

## Spatial and duplicate leakage

- **Closest work:** Karasiak et al. and Kattenborn et al. establish spatial
  dependence optimism in remote sensing; Barz and Denzler and the geospatial
  deduplication study establish duplicate contamination.
- **Not novel here:** hash deduplication, spatial blocking, or buffering alone.
- **Defensible delta:** integrate exact-hash, fixed-crop intersection, and
  unavailable-reference audits into one reliability-evaluation protocol.
  Outcome reversal is quantified only for the reference protocol and site
  pooling; duplicate and intersection checks are controls, not effect estimates.

## Measured-GSD validation

- **Closest work:** CRASAR-U-DROIDs supplies measured GSD and aligned UAS,
  crewed-aircraft and satellite products; operational disaster studies document
  GSD variability.
- **Not novel here:** paired cross-platform imagery or the CRASAR dataset.
- **Defensible delta:** evaluate received-image reliability on same-building
  UAS/post-event-satellite pairs across four sites and two events, with all
  evaluation sites excluded from classifier training and spatial-cluster
  uncertainty reported.

## Locked contribution boundary

The paper is a failure-finding audit and benchmark protocol. It does not
establish that the audited reference protocol is prevalent across published
remote-sensing practice, and it must not claim
that RCG, JS disagreement, ViM, KNN, spatial blocking, or hash deduplication is a
new primitive. Its defensible contribution is a replicated oracle--received
protocol gap, exact and spatial integrity controls, a reusable audit
implementation, and event-associated measured-GSD heterogeneity.
