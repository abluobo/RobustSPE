"""Scale calibration: mean-std and median-MAD with Bayesian prior."""
from __future__ import annotations

import numpy as np

from .config import FIBONACCI, MAD_SCALE


def nearest_fibonacci(value: float) -> int:
    return min(FIBONACCI, key=lambda f: abs(f - value))


def mad_scale(values: np.ndarray) -> float:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)) / MAD_SCALE)
    return max(mad, 1e-3)


def bayesian_median_mad(
    few_shot_sp: np.ndarray,
    med_global: float,
    mad_global: float,
    k: int,
    lam: float,
) -> tuple[float, float]:
    med_few = float(np.median(few_shot_sp))
    mad_few = mad_scale(few_shot_sp)
    mu_t = (k * med_few + lam * med_global) / (k + lam)
    sig_t = (k * mad_few + lam * mad_global) / (k + lam)
    return mu_t, sig_t


def bayesian_mean_std(
    few_shot_sp: np.ndarray,
    mu_global: float,
    sigma_global: float,
    k: int,
    lam: float,
) -> tuple[float, float]:
    mu_few = float(np.mean(few_shot_sp))
    sig_few = float(max(np.std(few_shot_sp), 1e-3))
    mu_t = (k * mu_few + lam * mu_global) / (k + lam)
    sig_t = (k * sig_few + lam * sigma_global) / (k + lam)
    return mu_t, sig_t


def restore_predictions(
    z_pred: np.ndarray,
    mu_t: float,
    sig_t: float,
    round_fibonacci: bool = True,
) -> np.ndarray:
    sp = z_pred * sig_t + mu_t
    if round_fibonacci:
        return np.array([nearest_fibonacci(p) for p in sp], dtype=float)
    return sp
