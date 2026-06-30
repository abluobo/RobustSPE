#!/usr/bin/env python3
"""K / λ ablation (Table 2): one model train per project, reuse z_pred."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn.metrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_spe.calibration import bayesian_median_mad, restore_predictions  # noqa: E402
from robust_spe.config import (  # noqa: E402
    K_VALUES,
    LAMBDA_VALUES,
    N_REPEAT,
    PROJECTS,
    RESULTS_DIR,
)
from robust_spe.data import global_robust_stats, source_projects  # noqa: E402
from robust_spe.model import get_device, train_zspace_regressor  # noqa: E402
from robust_spe.pipeline import build_target_tensors, build_zspace_training_data  # noqa: E402


def main() -> None:
    device = get_device()
    results = {f"K{k}_L{lam}": {} for k in K_VALUES for lam in LAMBDA_VALUES}
    t_start = time.time()

    print(f"Device: {device}")
    print(f"K: {K_VALUES}, λ: {LAMBDA_VALUES}, repeats: {N_REPEAT}")
    print("=" * 70)

    for idx, test_proj in enumerate(PROJECTS, 1):
        t0 = time.time()
        src = source_projects(test_proj)
        med_g, mad_g = global_robust_stats(src)

        x_tr, z_tr = build_zspace_training_data(src)
        x_te, y_te = build_target_tensors(test_proj)
        z_pred = train_zspace_regressor(x_tr, z_tr, x_te, device=device)

        rng = np.random.default_rng(42)
        n = len(y_te)
        max_k = max(K_VALUES)
        calib_sets = [rng.choice(n, min(max_k, n - 1), replace=False) for _ in range(N_REPEAT)]

        for k in K_VALUES:
            for lam in LAMBDA_VALUES:
                maes = []
                for rep in range(N_REPEAT):
                    calib = calib_sets[rep][:k]
                    mask = np.ones(n, dtype=bool)
                    mask[calib] = False
                    mu_t, sig_t = bayesian_median_mad(y_te[calib], med_g, mad_g, k, lam)
                    preds = restore_predictions(z_pred[mask], mu_t, sig_t)
                    maes.append(sklearn.metrics.mean_absolute_error(y_te[mask], preds))
                results[f"K{k}_L{lam}"][test_proj] = float(np.mean(maes))

        row = ", ".join(f"{results[f'K{k}_L10'][test_proj]:.3f}" for k in K_VALUES)
        print(f"[{idx:2d}/16] {test_proj}  K=[{row}]  {time.time() - t0:.0f}s", flush=True)

    print("\nK ablation (λ=10):")
    for k in K_VALUES:
        avg = np.mean(list(results[f"K{k}_L10"].values()))
        print(f"  K={k}: {avg:.3f}")

    print("\nλ ablation (K=20):")
    for lam in LAMBDA_VALUES:
        avg = np.mean(list(results[f"K20_L{lam}"].values()))
        print(f"  λ={lam}: {avg:.3f}")

    out = RESULTS_DIR / "results_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    print(f"Total: {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
