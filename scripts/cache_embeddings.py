#!/usr/bin/env python3
"""Cache mpnet (and optionally MiniLM) embeddings for all projects/splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_spe.config import DATA_SPLITS, EMBEDDINGS_DIR, PROJECTS  # noqa: E402

MODELS = {
    "mpnet": "all-mpnet-base-v2",
    "minilm": "all-MiniLM-L6-v2",
}


def cache_model(model_key: str) -> None:
    model_name = MODELS[model_key]
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {model_name} ({model_key}) ===")
    sbert = SentenceTransformer(model_name)
    for project in PROJECTS:
        for split in ("train", "val", "test"):
            out = EMBEDDINGS_DIR / f"{model_key}_{project}_{split}.npy"
            if out.exists():
                print(f"  {project}/{split} cached, skip")
                continue
            df = pd.read_csv(DATA_SPLITS / f"{project}_{split}.csv")
            texts = df["Issue"].fillna("").tolist()
            emb = sbert.encode(texts, batch_size=64, show_progress_bar=False)
            np.save(out, emb)
        print(f"  {project} done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache SBERT embeddings")
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        nargs="+",
        default=["mpnet"],
        help="Which embedding model(s) to cache (default: mpnet)",
    )
    args = parser.parse_args()
    for key in args.model:
        cache_model(key)
    print("\nDone.")


if __name__ == "__main__":
    main()
