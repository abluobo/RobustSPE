"""LOPO cross-project evaluation pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import sklearn.metrics

from .calibration import bayesian_mean_std, bayesian_median_mad, restore_predictions
from .config import (
    DEFAULT_K,
    DEFAULT_LAMBDA,
    N_REPEAT,
    PAPER_BASELINE,
    PROJECTS,
    RESULTS_DIR,
)
from .data import (
    global_mean_std,
    global_robust_stats,
    load_embeddings,
    load_story_points,
    per_project_z_stats,
    source_projects,
)
from .model import get_device, train_zspace_regressor

CalibMode = Literal["median_mad", "mean_std"]


def build_zspace_training_data(source_projs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_list, z_list = [], []
    for project in source_projs:
        mu, sigma = per_project_z_stats(project)
        for split in ("train", "val"):
            emb = load_embeddings(project, split)
            sp = load_story_points(project, split)
            x_list.append(emb)
            z_list.append((sp - mu) / sigma)
    return np.vstack(x_list), np.concatenate(z_list).astype(np.float32)


def build_target_tensors(test_project: str) -> tuple[np.ndarray, np.ndarray]:
    x_parts, y_parts = [], []
    for split in ("train", "val", "test"):
        x_parts.append(load_embeddings(test_project, split))
        y_parts.append(load_story_points(test_project, split))
    return np.vstack(x_parts), np.concatenate(y_parts)


def evaluate_calibration(
    z_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    mode: CalibMode,
    med_global: float,
    mad_global: float,
    mu_global: float,
    sigma_global: float,
    k: int,
    lam: float,
    n_repeat: int = N_REPEAT,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    maes = []
    for _ in range(n_repeat):
        calib_idx = rng.choice(n, min(k, n - 1), replace=False)
        eval_mask = np.ones(n, dtype=bool)
        eval_mask[calib_idx] = False
        few = y_true[calib_idx]
        if mode == "median_mad":
            mu_t, sig_t = bayesian_median_mad(few, med_global, mad_global, k, lam)
        else:
            mu_t, sig_t = bayesian_mean_std(few, mu_global, sigma_global, k, lam)
        preds = restore_predictions(z_pred[eval_mask], mu_t, sig_t)
        maes.append(sklearn.metrics.mean_absolute_error(y_true[eval_mask], preds))
    return float(np.mean(maes))


def run_lopo_project(
    test_project: str,
    *,
    k: int = DEFAULT_K,
    lam: float = DEFAULT_LAMBDA,
    n_repeat: int = N_REPEAT,
    device=None,
) -> dict[str, float]:
    src = source_projects(test_project)
    med_g, mad_g = global_robust_stats(src)
    mu_g, sig_g = global_mean_std(src)

    x_tr, z_tr = build_zspace_training_data(src)
    x_te, y_te = build_target_tensors(test_project)
    z_pred = train_zspace_regressor(x_tr, z_tr, x_te, device=device)

    robust_mae = evaluate_calibration(
        z_pred, y_te,
        mode="median_mad",
        med_global=med_g, mad_global=mad_g,
        mu_global=mu_g, sigma_global=sig_g,
        k=k, lam=lam, n_repeat=n_repeat,
    )
    musig_mae = evaluate_calibration(
        z_pred, y_te,
        mode="mean_std",
        med_global=med_g, mad_global=mad_g,
        mu_global=mu_g, sigma_global=sig_g,
        k=k, lam=lam, n_repeat=n_repeat,
    )
    return {
        "paper_baseline": PAPER_BASELINE[test_project],
        "musig_k20": musig_mae,
        "robust_k20": robust_mae,
    }


def run_full_evaluation(
    *,
    k: int = DEFAULT_K,
    lam: float = DEFAULT_LAMBDA,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    device = get_device()
    results = {}
    if verbose:
        print(f"Device: {device}")
        print(f"median+MAD calibration, K={k}, λ={lam}, {N_REPEAT} repeats")
        print("=" * 70)
    for i, project in enumerate(PROJECTS, 1):
        row = run_lopo_project(project, k=k, lam=lam, device=device)
        results[project] = row
        if verbose:
            b, ms, rob = row["paper_baseline"], row["musig_k20"], row["robust_k20"]
            print(f"[{i:2d}/16] {project:<22} baseline={b:.3f}  μ+σ={ms:.3f}  "
                  f"med+MAD={rob:.3f}  ({rob - b:+.3f})")
    if verbose:
        _print_summary(results)
    return results


def _print_summary(results: dict[str, dict[str, float]]) -> None:
    keys = ["paper_baseline", "musig_k20", "robust_k20"]
    avgs = {k: np.mean([results[p][k] for p in PROJECTS]) for k in keys}
    wins = sum(1 for p in PROJECTS if results[p]["robust_k20"] < results[p]["paper_baseline"])
    print("\n" + "=" * 70)
    print(f"Average MAE: baseline={avgs['paper_baseline']:.3f}  "
          f"μ+σ={avgs['musig_k20']:.3f}  med+MAD={avgs['robust_k20']:.3f}")
    print(f"Win rate vs baseline: {wins}/16")


def save_results(results: dict, path: Path | None = None) -> Path:
    path = path or (RESULTS_DIR / "results_robust_full.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path
