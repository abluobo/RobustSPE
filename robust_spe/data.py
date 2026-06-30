"""Data loading and global robust statistics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_SPLITS, EMBEDDINGS_DIR, EMBEDDING_PREFIX, PROJECTS


def split_csv(project: str, split: str) -> Path:
    return DATA_SPLITS / f"{project}_{split}.csv"


def load_story_points(project: str, split: str) -> np.ndarray:
    return pd.read_csv(split_csv(project, split))["Storypoint"].values.astype(np.float32)


def load_story_points_all(project: str, splits: tuple[str, ...] = ("train", "val", "test")) -> np.ndarray:
    return np.concatenate([load_story_points(project, s) for s in splits])


def embedding_path(project: str, split: str, prefix: str = EMBEDDING_PREFIX) -> Path:
    return EMBEDDINGS_DIR / f"{prefix}_{project}_{split}.npy"


def load_embeddings(project: str, split: str, prefix: str = EMBEDDING_PREFIX) -> np.ndarray:
    return np.load(embedding_path(project, split))


def source_projects(test_project: str) -> list[str]:
    return [p for p in PROJECTS if p != test_project]


def per_project_z_stats(project: str) -> tuple[float, float]:
    sp = load_story_points_all(project, ("train", "val"))
    return float(sp.mean()), float(max(sp.std(), 1e-3))


def global_robust_stats(source_projs: list[str]) -> tuple[float, float]:
    all_sp = np.concatenate([
        load_story_points_all(p, ("train", "val")) for p in source_projs
    ])
    med = float(np.median(all_sp))
    mad = float(np.median(np.abs(all_sp - np.median(all_sp))) / 0.675)
    return med, max(mad, 1e-3)


def global_mean_std(source_projs: list[str]) -> tuple[float, float]:
    all_sp = np.concatenate([
        load_story_points_all(p, ("train", "val")) for p in source_projs
    ])
    return float(all_sp.mean()), float(max(all_sp.std(), 1e-3))
