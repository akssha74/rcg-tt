# RCG-TT: Resolution Consistency Gate with Threshold Transfer

Code and result artifacts for the paper:

> **Resolution Consistency Gate with Threshold Transfer for Trust Collapse under
> Variable GSD in Disaster Remote Sensing**
> Akshay Sharma and Lalji Prasad, SAGE University, Indore, India.
> Submitted to *IEEE Transactions on Geoscience and Remote Sensing*, 2026.

Operational UAV and satellite imagery for post-disaster assessment does not arrive
at a fixed ground sampling distance (GSD). We show that scene/damage classifiers
undergo **resolution-induced trust collapse** — under a controlled GSD-proxy
resolution ladder, accuracy falls while confidence on errors stays high — and
propose **RCG-TT**, a multi-scale Jensen–Shannon consistency gate with
native-calibrated threshold transfer that detects these errors and preserves
auto-decision coverage under resolution change.

This repository lets a reviewer (a) verify that every table and figure in the
paper is backed by a hashed result artifact, and (b) regenerate all tables and
figures deterministically from those artifacts, without any dataset download.

---

## Repository layout

```
experiments/code/          Experiment + analysis scripts (Python)
experiments/derived/        Hashed result artifacts (JSON) + trained checkpoints (.pt)
experiments/raw/external/   Dataset manifests (pointers to public sources; no imagery)
paper/tables/               LaTeX tables used in the manuscript
paper/figures/              Figures (PDF for LaTeX, PNG previews)
evidence/                   Artifact, claim, and citation ledgers (provenance)
scripts/                    reproduce.sh, verify_hashes.py, make_manifest.py
RELEASE_SHA256SUMS.txt      SHA-256 of every file in this distribution
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Integrity: re-hash artifacts and check them against the paper ledger
python3 scripts/verify_hashes.py

# 2) Regenerate all tables/figures from the derived artifacts and diff vs the paper
bash scripts/reproduce.sh
```

`scripts/reproduce.sh` needs only `numpy` + `matplotlib` (no GPU, no datasets) and
finishes in seconds, because it packages the already-computed result JSON into the
paper's LaTeX tables and figures.

## Two levels of reproduction

**(A) Deterministic packaging — default.** `experiments/derived/*.json` are the
computed results (AUROC, coverage, lifts, per-operator/per-γ sweeps). The scripts
`generate_paper_artifacts.py`, `generate_transfer_figures.py`, and
`run_immune_packaging.py` turn them into the exact LaTeX tables in `paper/tables/`.
`reproduce.sh` regenerates and diffs these; tables reproduce byte-for-byte.

**(B) Full re-computation — optional, needs data + GPU/MPS.** To recompute from
raw imagery:

- **AIDER** (5 classes): obtain the public dataset (see
  `experiments/raw/external/AIDER_manifest.txt`), set `AIDER_ROOT` to its path, then

  ```bash
  export AIDER_ROOT=/path/to/AIDER
  python3 experiments/code/run_aider_resolution.py --aider-root "$AIDER_ROOT" --work experiments/derived/aider_rcg
  python3 experiments/code/run_confirmatory.py     # degraded-scale RCG vs confidence
  python3 experiments/code/baseline_compare_s8.py  # temp-scaling / energy / ensembles
  ```

  The released `experiments/derived/aider_rcg/aider_splits.json` pins the exact
  train/val/test split (dataset-relative image paths), so evaluation is
  reproducible without redistributing the imagery. Trained checkpoints
  (`model_*.pt`) are included so you can re-evaluate without retraining.

- **Satellite Images of Hurricane Damage**: loaded from Hugging Face
  (`jonathan-roberts1/Satellite-Images-of-Hurricane-Damage`) by
  `run_hurricane_rcg.py` / `run_strengthen_unambiguous.py`.

- **xBD** (negative control): obtain from https://xview2.org/dataset .

## Datasets and licensing

No third-party imagery is redistributed here. AIDER, Satellite Images of Hurricane
Damage, and xBD are public but governed by their own licenses — obtain them from
the official sources listed in `experiments/raw/external/`. The code and the
derived result artifacts in this repository are released under the MIT License
(see `LICENSE`).

## Provenance and integrity

`evidence/artifact-ledger.jsonl` maps every paper table/figure to (i) the script
that generated it, (ii) its source artifact(s), and (iii) a SHA-256 digest.
`scripts/verify_hashes.py` recomputes and checks these; a clean run proves the
files here are the ones behind the published numbers.

Two documented, intentional exceptions:
- `aider_splits.json` was path-normalized for release (local absolute image paths
  → dataset-relative paths); the verifier reports it as `NORMAL`.
- Figure **PDFs** are not byte-reproducible across matplotlib/font builds; integrity
  is guaranteed for the data artifacts and the LaTeX tables, and the ledger records
  the digests of the distributed figure files.

## Citation

See `CITATION.cff`.
