# Feasibility addendum: leakage-free measured-GSD replication v2

Status: **FROZEN before model training or evaluation**  
Date: 2026-07-24

The original preparation completed guarded crops for Lancaster Canyon Gate and
Summerlin San Carlos, then spent more than one hour on an unauthenticated
multi-gigabyte third-site download without completing it. The process was
terminated with exit code 143. No model was trained and no validation or test
outcome was evaluated.

To keep the experiment locally feasible, the fixed training set is replaced by
the two completed, label-rich UAS sites:

- `090403-Lancaster-Canyon-Gate.geo.tif` (Hurricane Harvey);
- `1001-Summerlin-San-Carlos.geo.tif` (Hurricane Ian).

Together they provide 1,628 guarded labelled crops before split filtering is
summarised. The 2048-pixel block assignment, 256-pixel guard, exhaustive
intersection audit, ResNet-18 seeds, 12-epoch training, metrics, held-out
four-site/two-event paired evaluation, bootstrap, and W10--W13 criteria are
unchanged. Harlem Heights, McGregor College Parkway, and both Mexico Beach sites
remain excluded from training and model selection.
