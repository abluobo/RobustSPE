"""
全量 16 项目评估：median+MAD 稳健校准
预计耗时 ~8 小时

方法：
  训练：各源项目 z-score 归一化，消除量纲差异
  推理：K=20 带标注样本 + Bayesian 混合，用 median+MAD 代替 mean+std

  mu_T  = (K * median_few  + lambda * median_global) / (K + lambda)
  sig_T = (K * MAD_few/0.675 + lambda * MAD_global/0.675) / (K + lambda)

对比：论文基线（hil-se SBERT-Regression MAE）
"""
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
import json, time

DATA_DIR  = "../Data/GPT2SP Data/Split/"
CACHE_DIR = "../Data/Embeddings/"
DATAS = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement",
         "duracloud", "jirasoftware", "mesos", "moodle", "mule", "mulestudio",
         "springxd", "talenddataquality", "talendesb", "titanium", "usergrid"]
PAPER_BASELINE = {
    "appceleratorstudio": 2.50, "aptanastudio": 4.49, "bamboo": 1.91,
    "clover": 3.60, "datamanagement": 7.41, "duracloud": 2.71,
    "jirasoftware": 2.79, "mesos": 1.91, "moodle": 12.21, "mule": 2.69,
    "mulestudio": 3.38, "springxd": 2.11, "talenddataquality": 3.54,
    "talendesb": 1.88, "titanium": 3.32, "usergrid": 1.56,
}
PREV_MUSIG = {  # 原始 μ+σ 全量结果（供对比）
    "appceleratorstudio": 2.203, "aptanastudio": 4.089, "bamboo": 1.206,
    "clover": 3.102, "datamanagement": 9.573, "duracloud": 1.276,
    "jirasoftware": 2.301, "mesos": 1.518, "moodle": 11.543, "mule": 2.530,
    "mulestudio": 3.376, "springxd": 2.013, "talenddataquality": 3.485,
    "talendesb": 1.204, "titanium": 3.171, "usergrid": 1.017,
}
EPOCHS     = 1000
BATCH_SIZE = 32
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
FIBS   = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
LAMBDA = 10
K      = 20
N_REPEAT = 10
RESULT_FILE = "../results_robust_full.json"

def nearest_fib(x):
    return min(FIBS, key=lambda f: abs(f - x))

def load_sp(name, split):
    return pd.read_csv(f"{DATA_DIR}{name}_{split}.csv")["Storypoint"].values.astype(np.float32)

class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)

results = {}
start_time = time.time()

print(f"设备: {DEVICE}")
print(f"方法: median+MAD Bayesian 校准，K={K}，λ={LAMBDA}，{N_REPEAT} 次重复")
print("=" * 70)

for proj_idx, test_proj in enumerate(DATAS):
    t0 = time.time()
    print(f"\n[{proj_idx+1:2d}/16] {test_proj}", flush=True)
    src_projs = [d for d in DATAS if d != test_proj]

    # ── 统计量 ──────────────────────────────────────────────────────
    proj_stats = {}
    for p in src_projs:
        sp = np.concatenate([load_sp(p, s) for s in ["train", "val"]])
        proj_stats[p] = (float(sp.mean()), float(max(sp.std(), 1e-3)))

    # 稳健全局参照（所有源项目拼合）
    all_src_sp = np.concatenate([
        np.concatenate([load_sp(p, s) for s in ["train", "val"]]) for p in src_projs
    ])
    med_global = float(np.median(all_src_sp))
    mad_global = float(max(np.median(np.abs(all_src_sp - np.median(all_src_sp))) / 0.675, 1e-3))

    # ── 构造 z-score 训练数据 ───────────────────────────────────────
    X_list, z_list = [], []
    for p in src_projs:
        mu, sigma = proj_stats[p]
        for s in ["train", "val"]:
            emb = np.load(f"{CACHE_DIR}mpnet_{p}_{s}.npy")
            sp  = load_sp(p, s)
            X_list.append(emb)
            z_list.append((sp - mu) / sigma)

    X_tr = np.vstack(X_list)
    y_tr = np.concatenate(z_list).astype(np.float32)
    X_te = np.vstack([np.load(f"{CACHE_DIR}mpnet_{test_proj}_{s}.npy")
                      for s in ["train", "val", "test"]])
    y_te = np.concatenate([load_sp(test_proj, s) for s in ["train", "val", "test"]])

    # ── 训练模型 ────────────────────────────────────────────────────
    torch.manual_seed(0); np.random.seed(0)
    idx = np.random.permutation(len(X_tr))
    X_tr_t = torch.tensor(X_tr[idx], dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_tr[idx], dtype=torch.float32).to(DEVICE)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

    model = Linear(X_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.3, min_lr=1e-6)
    loss_fn = nn.L1Loss()

    best_loss, best_state = float("inf"), None
    n = len(X_tr_t)
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for s in range(0, n, BATCH_SIZE):
            xb, yb = X_tr_t[s:s+BATCH_SIZE], y_tr_t[s:s+BATCH_SIZE]
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= n
        sched.step(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        z_pred = model(X_te_t).cpu().numpy()

    # ── median+MAD Bayesian 校准（N_REPEAT 次）──────────────────────
    rng = np.random.default_rng(42)
    N = len(y_te)
    maes = []
    for _ in range(N_REPEAT):
        calib_idx = rng.choice(N, min(K, N - 1), replace=False)
        eval_mask = np.ones(N, dtype=bool); eval_mask[calib_idx] = False

        med_few = float(np.median(y_te[calib_idx]))
        mad_few = float(max(np.median(np.abs(y_te[calib_idx] - med_few)) / 0.675, 1e-3))
        mu_T    = (K * med_few + LAMBDA * med_global) / (K + LAMBDA)
        sig_T   = (K * mad_few + LAMBDA * mad_global) / (K + LAMBDA)

        sp_pred   = z_pred[eval_mask] * sig_T + mu_T
        preds_fib = np.array([nearest_fib(p) for p in sp_pred])
        maes.append(sklearn.metrics.mean_absolute_error(y_te[eval_mask], preds_fib))

    mae_robust = float(np.mean(maes))
    b = PAPER_BASELINE[test_proj]
    prev = PREV_MUSIG[test_proj]
    print(f"  基线={b:.3f}  μ+σ={prev:.3f}  med+MAD={mae_robust:.3f}  "
          f"({mae_robust - b:+.3f} vs 基线)  耗时 {time.time()-t0:.0f}s", flush=True)

    results[test_proj] = {
        "paper_baseline": b,
        "musig_k20":      prev,
        "robust_k20":     mae_robust,
    }

# ── 汇总 ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"{'项目':<22} {'基线':>7} {'μ+σ K20':>9} {'med+MAD':>9} {'改善':>8} {'vs μ+σ':>8}")
for proj in DATAS:
    r = results[proj]
    b, ms, rob = r["paper_baseline"], r["musig_k20"], r["robust_k20"]
    print(f"{proj:<22} {b:>7.3f} {ms:>9.3f} {rob:>9.3f} {rob-b:>+8.3f} {rob-ms:>+8.3f}")

avgs = {k: np.mean([results[p][k] for p in DATAS])
        for k in ["paper_baseline", "musig_k20", "robust_k20"]}
print(f"\n{'平均':<22} {avgs['paper_baseline']:>7.3f} {avgs['musig_k20']:>9.3f} "
      f"{avgs['robust_k20']:>9.3f} {avgs['robust_k20']-avgs['paper_baseline']:>+8.3f} "
      f"{avgs['robust_k20']-avgs['musig_k20']:>+8.3f}")

wins = sum(1 for p in DATAS if results[p]["robust_k20"] < results[p]["paper_baseline"])
print(f"\n胜率（vs 基线）: {wins}/16")
print(f"总耗时: {(time.time()-start_time)/3600:.1f} 小时")

with open(RESULT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"结果已保存至: {RESULT_FILE}")
