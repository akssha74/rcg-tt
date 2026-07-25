#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

cd "$ROOT"

"$PYTHON" - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
manifest_path = root / "experiments/imported/import_manifest.json"
for row in json.loads(manifest_path.read_text()):
    path = root / row["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        raise RuntimeError(f"hash mismatch: {row['path']}")
print("IMPORTED_EVIDENCE_VERIFIED")
PY

"$PYTHON" experiments/code/adjudicate_paired_protocol.py
"$PYTHON" experiments/code/verify_audit_claims.py
"$PYTHON" experiments/code/verify_per_example_outputs.py
"$PYTHON" experiments/code/generate_audit_artifacts.py
"$PYTHON" experiments/code/verify_submission_package.py

echo "RESOLUTION_AUDIT_REPRODUCTION_COMPLETE"
