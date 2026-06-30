#!/usr/bin/env bash
# Reproduce main paper results (Table 1 + significance).
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

echo ">>> Caching mpnet embeddings (skip if already present)..."
python scripts/cache_embeddings.py --model mpnet

echo ">>> Running main LOPO evaluation (~1.5 h)..."
python scripts/run_main.py

echo ">>> Significance tests..."
python scripts/run_significance.py

echo ">>> Done. Results: results/results_robust_full.json"
