# Research brief

## Theme (related to and supporting the synopsis)

Information-set fairness and leakage controls for uncertainty estimates under
spatial-resolution shift in disaster remote sensing.

## Problem and scope

Resolution-consistency scores can look exceptionally effective when a synthetic
benchmark lets them compare a coarse test view with the unavailable original
fine image. This creates a privileged-information advantage over confidence and
feature-space baselines. Existing evaluations can also be distorted by exact
cross-split duplicates and overlapping geospatial crops.

## Research question

How much do unavailable reference views and split leakage inflate apparent
reliability under resolution shift, and which deployable scores remain useful
under matched received-image information and measured real-world GSD?

## Falsifiable hypothesis

Privileged-view consistency will show significantly higher error AUROC than its
received-image counterpart on both AIDER and Hurricane across three seeds. The
inflation will remain detectable after exact-hash correction, while leakage-free
multi-event CRASAR evaluation will produce materially different and more
uncertain operational conclusions than the contaminated two-site analysis.

## Intended contribution

A failure-finding audit protocol, reproducible implementation, and multi-corpus
evidence that separate reference-dependent diagnostics from deployable
reliability scores. The study will compare confidence, energy, EO-kNN, ViM, and
received-image consistency under matched information and publish all null and
negative results.

## Delta from author's prior work (do-not-reuse set)

The archived resolution-reliability study asserted a winning RCG and failed a
separate VLM clause. The RCG-GSD study later exposed privileged-view access,
cross-split contamination, and failed deployable gates. This sibling treats
those failures as the object of study, supersedes contaminated endpoints, and
adds matched ViM/KNN baselines, guarded spatial splits, and multi-event paired
measured-GSD inference.

## Available artifacts and resources

Apple M4 Max with 64 GB memory and MPS; public AIDER, Hurricane Damage, xBD, and
CRASAR-U-DROIDs data; six primary ResNet checkpoints; prior failed W8/W9
artifacts; local LaTeX and Python environments. Imported evidence must retain
its source path and SHA-256.

## Primary metric and meaningful effect

Paired seed-level AUROC difference between privileged-view and received-image
consistency for error detection. A meaningful primary effect is a positive mean
on both corpora with paired bootstrap 95% intervals excluding zero. Secondary
metrics are AUPRC, FPR95, realised selective coverage/risk, leakage counts, and
paired measured-GSD site-cluster intervals.

## What would falsify the idea

No reproducible inflation on either primary corpus; inability to produce
leakage-free measured-GSD evidence with a valid classifier; or discovery of an
existing remote-sensing paper that already performs the same combined audit.

## Risks, ethics, privacy, and licensing

No human subjects or new annotation. Public imagery may depict disaster damage
but analysis is building-level classification without identity inference.
Respect dataset licences, record snapshots and hashes, and separate code and
artifact licences. The main scientific risk is turning a post-hoc failure into
an overstated novelty claim; preregistration and independent default-reject
review are mandatory.

## Target venue and constraints

International Journal of Remote Sensing research article, UK English, IJRS
author-year references and disclosure sections. All computation is local; no
paid APIs or external compute. Submission remains blocked until strict
validation and two independent agent-only default-reject reviews pass.
