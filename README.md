# RCG-TT: Resolution Consistency Gate with Threshold Transfer

Code and result artifacts for the paper:

> **Resolution Consistency Gate with Threshold Transfer for Trust Collapse under
> Variable GSD in Disaster Remote Sensing**
> Akshay Sharma and Lalji Prasad, SAGE University, Indore, India.
> Prepared for the *International Journal of Remote Sensing*, 2026.

Operational UAV and satellite imagery for post-disaster assessment does not arrive
at a fixed ground sampling distance (GSD). We show that scene/damage classifiers
undergo **resolution-induced trust collapse** — under a controlled GSD-proxy
resolution ladder, accuracy falls while confidence on errors stays high — and
propose **RCG-TT**, a multi-scale Jensen–Shannon consistency gate with
native-calibrated threshold transfer that detects these errors and preserves
auto-decision coverage under resolution change.

This repository lets a reviewer (a) verify that each table and figure in the
paper is backed by a hashed result artifact, and (b) regenerate all tables and
figures from those artifacts, without any dataset download.
The final strengthening release includes three independently trained primary
ResNet-18 seeds per corpus, an EO-specific k-nearest-neighbour OOD comparison,
and 125 paired operational UAS/satellite buildings across two sites at measured
GSD.

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
requirements-full.txt       pinned environment for training/re-evaluation
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

**(A) Artifact verification and packaging — default.** `experiments/derived/*.json` are the
computed results (AUROC, coverage, lifts, per-operator/per-γ sweeps). The scripts
`generate_paper_artifacts.py`, `generate_transfer_figures.py`, and
`run_immune_packaging.py`, followed by `generate_revision_tables.py`, turn them
into the legacy LaTeX assets; `generate_strengthening_artifacts.py` packages the
multi-seed, EO-OOD, and measured-GSD results.
`reproduce.sh` regenerates and diffs these; tables reproduce byte-for-byte.

**(B) Checkpoint re-evaluation / full retraining — optional, needs data +
GPU/MPS.** The release contains all AIDER, Hurricane, and measured-GSD primary
seed checkpoints plus score arrays and histories. Install the full environment:

```bash
pip install -r requirements-full.txt
```

- **AIDER** (5 classes): obtain the public dataset (see
  `experiments/raw/external/AIDER_manifest.txt`), set `AIDER_ROOT` to its path, then

  ```bash
  export AIDER_ROOT=/path/to/AIDER
  python3 experiments/code/run_aider_resolution.py --aider-root "$AIDER_ROOT" --work experiments/derived/aider_rcg
  ```

  The released `experiments/derived/aider_rcg/aider_splits.json` pins the exact
  train/val/test split (dataset-relative image paths), so evaluation is
  reproducible without redistributing the imagery. Trained checkpoints
  (`model_*.pt`) are included so the released checkpoints can be re-evaluated.

- **Satellite Images of Hurricane Damage**: the exact Hugging Face source
  (`jonathan-roberts1/Satellite-Images-of-Hurricane-Damage`) is recorded in
  `experiments/raw/external/HurricaneDamage_manifest.txt`; the released
  checkpoints and summaries are under
  `experiments/derived/greatness_strengthening/hurricane/`.

- **xBD** (negative control): obtain from https://xview2.org/dataset .

- **CRASAR-U-DROIDs measured GSD**: source snapshot, selected orthomosaics,
  GSDs, and licence are fixed in `experiments/raw/external/CRASAR_manifest.txt`.
  Imagery is fetched from Hugging Face by `run_measured_gsd_crasar.py`; no
  third-party image patches are redistributed.

Primary strengthening commands:

```bash
export AIDER_ROOT=/path/to/AIDER
python3 experiments/code/run_primary_multiseed.py --datasets aider hurricane
python3 experiments/code/run_measured_gsd_crasar.py --epochs 5
python3 experiments/code/run_paired_measured_gsd.py
python3 experiments/code/run_multisite_paired_gsd.py
python3 experiments/code/generate_strengthening_artifacts.py
```

## Datasets and licensing

No third-party imagery is redistributed here. AIDER, Satellite Images of Hurricane
Damage, xBD, and CRASAR-U-DROIDs are public but governed by their own licenses —
obtain them from the official sources listed in `experiments/raw/external/`.

Repository licensing is split by content type: source code under
`experiments/code/` and `scripts/` is MIT-licensed; generated figures, tables,
result summaries, trained checkpoints, provenance ledgers, and documentation are
licensed under CC BY 4.0. See `LICENSING.md` and `LICENSE`.

## Provenance and integrity

`evidence/artifact-ledger.jsonl` maps every paper table/figure to (i) the script
that generated it, (ii) its source artifact(s), and (iii) a SHA-256 digest.
`scripts/verify_hashes.py` recomputes and checks these; a clean run proves the
files here are the ones behind the submitted numbers.

Two documented, intentional exceptions:
- `aider_splits.json` was path-normalized for release (local absolute image paths
  → dataset-relative paths); the verifier reports it as `NORMAL`.
- Figure **PDFs** are not byte-reproducible across matplotlib/font builds; integrity
  is guaranteed for the data artifacts and the LaTeX tables, and the ledger records
  the digests of the distributed figure files.

## Citation

See `CITATION.cff`.
