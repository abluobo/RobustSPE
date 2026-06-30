"""
SBERT-Regression Cross-Project 复现脚本
原代码: SBERT_Regression_Cross.py (TensorFlow Keras)
本脚本: 等价 PyTorch 实现（单线性层 + Adam + MAE loss，1000 epochs）
目标: 复现论文报告的平均 MAE 3.63
"""
import numpy as np
import scipy.stats
import sklearn.metrics
import pandas as pd
import torch
import torch.nn as nn
import time
import os

DATA_DIR = "../Data/GPT2SP Data/Split/"
RESULTS_DIR = "../Results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATAS = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement",
         "duracloud", "jirasoftware", "mesos", "moodle", "mule", "mulestudio",
         "springxd", "talenddataquality", "talendesb", "titanium", "usergrid"]

ITRS = 10
EPOCHS = 1000
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"使用设备: {DEVICE}")

# ── SBERT 编码 ────────────────────────────────────────────────────────
print("加载 SBERT 模型...")
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer("all-MiniLM-L6-v2")

def load_split(name, split):
    df = pd.read_csv(f"{DATA_DIR}{name}_{split}.csv")
    return df

def encode_df(df):
    texts = df["Issue"].fillna("").tolist()
    return sbert.encode(texts, batch_size=64, show_progress_bar=False)

# 预先编码所有项目所有split，避免重复计算
print("预编码所有数据（首次运行约 5 分钟）...")
cache = {}
for d in DATAS:
    for split in ["train", "val", "test"]:
        df = load_split(d, split)
        emb = encode_df(df)
        sp = df["Storypoint"].values.astype(np.float32)
        cache[(d, split)] = (emb, sp)
    print(f"  {d} done")

# ── 模型（单线性层，等价于 Keras Dense(1, activation='linear')）────────
class LinearReg(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)

def train_and_test(test_name, itr):
    # 训练集：所有其他项目的 train + val
    X_tr_list, y_tr_list = [], []
    for d in DATAS:
        if d != test_name:
            for split in ["train", "val"]:
                emb, sp = cache[(d, split)]
                X_tr_list.append(emb)
                y_tr_list.append(sp)
    X_tr = np.vstack(X_tr_list)
    y_tr = np.concatenate(y_tr_list)

    # 测试集：目标项目的 train + val + test（与原代码一致）
    X_te_list, y_te_list = [], []
    for split in ["train", "val", "test"]:
        emb, sp = cache[(test_name, split)]
        X_te_list.append(emb)
        y_te_list.append(sp)
    X_te = np.vstack(X_te_list)
    y_te = np.concatenate(y_te_list)

    dim = X_tr.shape[1]
    results = []

    for i in range(itr):
        idx = np.random.permutation(len(X_tr))
        X_tr_s = torch.tensor(X_tr[idx], dtype=torch.float32).to(DEVICE)
        y_tr_s = torch.tensor(y_tr[idx], dtype=torch.float32).to(DEVICE)
        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

        model = LinearReg(dim).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.L1Loss()  # MAE loss，与原代码一致

        # ReduceLROnPlateau（patience=10, factor=0.3）
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.3, min_lr=1e-6
        )

        best_loss = float("inf")
        best_state = None

        model.train()
        n = len(X_tr_s)
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for start in range(0, n, BATCH_SIZE):
                xb = X_tr_s[start:start+BATCH_SIZE]
                yb = y_tr_s[start:start+BATCH_SIZE]
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= n
            scheduler.step(epoch_loss)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds_te = model(X_te_t).cpu().numpy()
            preds_tr = model(X_tr_s).cpu().numpy()

        mae_te = sklearn.metrics.mean_absolute_error(y_te, preds_te)
        mae_tr = sklearn.metrics.mean_absolute_error(y_tr, preds_tr)
        r_te = scipy.stats.pearsonr(preds_te, y_te)[0]
        rs_te = scipy.stats.spearmanr(preds_te, y_te).statistic

        results.append((mae_tr, mae_te, r_te, rs_te))
        print(f"  [{test_name}] itr {i+1}/{itr}  MAE_test={mae_te:.3f}")

    return results

# ── 主循环 ─────────────────────────────────────────────────────────────
all_results = []
times = []

for d in ["mesos"]:  # 先跑一个验证，论文期望值 1.91
    print(f"\n=== {d} ===")
    t0 = time.time()
    res = train_and_test(d, ITRS)
    elapsed = (time.time() - t0) / ITRS
    times.append(elapsed)

    for mae_tr, mae_te, r_te, rs_te in res:
        all_results.append({
            "Data": d,
            "MAE Train": mae_tr,
            "MAE Test": mae_te,
            "Pearson Test": r_te,
            "Spearman Test": rs_te,
        })

df_res = pd.DataFrame(all_results)
df_res.to_csv(f"{RESULTS_DIR}SBERT-Regression-Cross-Project-Reproduced.csv", index=False)

# 按项目取平均
per_proj = df_res.groupby("Data")["MAE Test"].mean()
print("\n===== 复现结果 =====")
print(per_proj.to_string())
print(f"\n平均 MAE: {per_proj.mean():.3f}  (论文报告: 3.63)")
print(f"\n总耗时: {sum(times)*ITRS/60:.1f} 分钟")
