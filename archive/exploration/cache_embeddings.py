"""
一次性编码并缓存所有 SBERT embeddings 到磁盘
之后每次实验直接加载，无需重新编码
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

DATA_DIR = "../Data/GPT2SP Data/Split/"
CACHE_DIR = "../Data/Embeddings/"
os.makedirs(CACHE_DIR, exist_ok=True)

DATAS = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement",
         "duracloud", "jirasoftware", "mesos", "moodle", "mule", "mulestudio",
         "springxd", "talenddataquality", "talendesb", "titanium", "usergrid"]

for model_name, short in [("all-MiniLM-L6-v2", "minilm"), ("all-mpnet-base-v2", "mpnet")]:
    print(f"\n=== {model_name} ===")
    sbert = SentenceTransformer(model_name)
    for d in DATAS:
        for split in ["train", "val", "test"]:
            path = f"{CACHE_DIR}{short}_{d}_{split}.npy"
            if os.path.exists(path):
                print(f"  {d}/{split} 已缓存，跳过")
                continue
            df = pd.read_csv(f"{DATA_DIR}{d}_{split}.csv")
            emb = sbert.encode(df["Issue"].fillna("").tolist(), batch_size=64, show_progress_bar=False)
            np.save(path, emb)
        print(f"  {d} done")

print("\n全部缓存完成！")
