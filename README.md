# Resolution Reliability Audit

Reproducibility package for:

> **When resolution consistency fails: an information-matched audit for
> disaster remote sensing**

## Main finding

A reference-dependent resolution-consistency diagnostic appears much stronger
than a received-image-only score on AIDER and Hurricane Damage. The paper
reports this as an oracle–received protocol gap, not as a causal estimate of
hidden-information inflation. A post-hoc anchor-matched sensitivity controls
the score anchor and comparison count and retains the gap.

The measured-GSD analysis uses 1,441 paired UAS/satellite buildings across four
acquisitions and two hurricane events. W11 and the preregistered mean-based W13
pass; W12 and the post-hoc all-seed W13 robustness sensitivity fail. The
negative and superseded results are retained.

## Reproduce the registered paper

Requirements:

- Python 3.10
- packages in `requirements-lock.txt`
- Tectonic 0.16.9

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
bash reproduce.sh
```

This verifies every imported artifact hash, recomputes registered headline
checks and protocol adjudication, regenerates tables/figures, and clean-builds
the IJRS manuscript.

Full classifier retraining requires the public AIDER, Hurricane Damage and
CRASAR-U-DROIDs datasets documented in `research/data-sources.jsonl`. The
provided checkpoints and per-example score arrays support checkpoint-level
reproduction without redistributing third-party imagery.

## Package map

- `paper/` — self-contained IJRS source and compiled paper
- `experiments/code/` — audit, verification and packaging code
- `experiments/imported/` — hashed corrective runs and checkpoints
- `experiments/derived/` — adjudication, metadata and verification outputs
- `evidence/` — claim, citation and artifact ledgers
- `research/` — locked contract, preregistration and analysis deviations
- `reviews/` — independent review state
- `submission/` — deterministic source/reproducibility archives and manifests

## Licensing

See `LICENSES.md`. Code is MIT; original research artefacts and documentation
are CC BY 4.0. Third-party data and weights retain their source licences.
