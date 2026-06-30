#!/usr/bin/env python3
"""Wilcoxon significance tests and failure-project analysis."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_spe.config import DATA_SPLITS, PROJECTS, RESULTS_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=RESULTS_DIR / "results_robust_full.json",
    )
    args = parser.parse_args()

    with open(args.input) as f:
        results = json.load(f)

    baseline = np.array([results[p]["paper_baseline"] for p in PROJECTS])
    musig = np.array([results[p]["musig_k20"] for p in PROJECTS])
    robust = np.array([results[p]["robust_k20"] for p in PROJECTS])

    print("=" * 60)
    print("Wilcoxon signed-rank (one-sided: med+MAD < baseline)")
    print("=" * 60)

    stat, p = stats.wilcoxon(robust, baseline, alternative="less")
    print(f"\nmed+MAD vs baseline: W={stat:.1f}, p={p:.4f}")

    stat2, p2 = stats.wilcoxon(musig, baseline, alternative="less")
    print(f"μ+σ vs baseline:     W={stat2:.1f}, p={p2:.4f}")

    stat3, p3 = stats.wilcoxon(robust, musig)
    print(f"med+MAD vs μ+σ:      W={stat3:.1f}, p={p3:.4f} (two-sided)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats.wilcoxon(robust, baseline, alternative="less", method="exact")
    z = stats.norm.ppf(p)
    r = abs(z) / np.sqrt(len(PROJECTS))
    print(f"\nEffect size r ≈ {r:.3f}")

    fail = [p for p in PROJECTS if results[p]["robust_k20"] > results[p]["paper_baseline"]]
    win = [p for p in PROJECTS if p not in fail]
    print("\n" + "=" * 60)
    print(f"Failures ({len(fail)}): {fail}")
    print(f"Wins ({len(win)}): {len(win)} projects")

    print(f"\n{'Project':<22} {'Mean':>7} {'Median':>7} {'Std':>7} {'MAD':>7} {'Skew':>7} {'Δ MAE':>9}")
    for project in PROJECTS:
        sp = []
        for split in ("train", "val", "test"):
            df = pd.read_csv(DATA_SPLITS / f"{project}_{split}.csv")
            sp.extend(df["Storypoint"].tolist())
        sp = np.array(sp, dtype=float)
        med = np.median(sp)
        mad = np.median(np.abs(sp - med)) / 0.675
        delta = results[project]["robust_k20"] - results[project]["paper_baseline"]
        flag = "FAIL" if project in fail else "OK"
        print(f"{project:<22} {sp.mean():>7.2f} {med:>7.2f} {sp.std():>7.2f} {mad:>7.2f} "
              f"{stats.skew(sp):>7.2f} {delta:>+9.3f} {flag}")


if __name__ == "__main__":
    main()
