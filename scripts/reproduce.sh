#!/usr/bin/env bash
# Reproduce the paper's tables and figures from the released derived artifacts.
#
# There are two levels of reproduction:
#
#   (A) DETERMINISTIC PACKAGING (default, no datasets, no GPU, seconds):
#       regenerate every LaTeX table and figure from the derived result JSON in
#       experiments/derived/ and confirm they match the committed paper assets.
#
#   (B) CHECKPOINT RE-EVALUATION (optional, requires datasets + a GPU/MPS):
#       evaluate released checkpoints where code is provided. See the README.
#
# Usage:
#   bash scripts/reproduce.sh            # packaging + integrity check
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

echo "=== [1/3] Integrity check against the paper ledger ==="
python3 scripts/verify_hashes.py

echo
echo "=== [2/3] Regenerating tables/figures from derived artifacts ==="
# Snapshot current committed assets, regenerate, then diff.
TMP="$(mktemp -d)"
cp -r paper/tables "$TMP/tables_committed"
cp -r paper/figures "$TMP/figures_committed"

python3 experiments/code/generate_paper_artifacts.py
python3 experiments/code/generate_transfer_figures.py
python3 experiments/code/run_immune_packaging.py
python3 experiments/code/generate_revision_tables.py

echo
echo "=== [3/3] Diffing regenerated LaTeX tables vs committed ==="
status=0
for f in paper/tables/*.tex; do
  if ! diff -q "$TMP/tables_committed/$(basename "$f")" "$f" >/dev/null 2>&1; then
    echo "CHANGED  $f"
    status=1
  else
    echo "MATCH    $f"
  fi
done
if [ "$status" -eq 0 ]; then
  echo "All tables reproduced byte-for-byte."
else
  echo "Some tables changed (inspect diffs above)."
fi
echo "(Figures are non-deterministic in byte layout across matplotlib/font builds;"
echo " compare visually against paper/figures/*.png if needed.)"
rm -rf "$TMP"
exit "$status"
