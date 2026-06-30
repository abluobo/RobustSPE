#!/usr/bin/env python3
"""Reproduce main paper results (Table 1): 16-project LOPO, median+MAD calibration."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_spe.config import DEFAULT_K, DEFAULT_LAMBDA, RESULTS_DIR  # noqa: E402
from robust_spe.pipeline import run_full_evaluation, save_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, default=DEFAULT_K)
    parser.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=RESULTS_DIR / "results_robust_full.json",
    )
    args = parser.parse_args()

    t0 = time.time()
    results = run_full_evaluation(k=args.K, lam=args.lam)
    out = save_results(results, args.output)
    print(f"\nSaved: {out}")
    print(f"Elapsed: {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
