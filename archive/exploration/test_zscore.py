"""
方法：Per-Project Z-score 归一化 + 相似度加权推理校准

训练：
  对每个源项目，将 SP 做 z-score 归一化 (SP - μ_p) / σ_p
  模型学习预测无量纲的 z-score（各项目 scale 对齐）

推理：
  预测出 z 后，需要还原为 SP = z * σ_T + μ_T
  目标项目 μ_T、σ_T 未知，用源项目加权估算：
    - 权重 = 目标项目 embedding 与各源项目 embedding 的余弦相似度
    - μ_T_est = Σ(w_i * μ_i),  σ_T_est = Σ(w_i * σ_i)

对比多种校准策略：
  ZS-global:  用全体源项目的全局 μ/σ（不做相似度加权）
  ZS-sim:     相似度加权校准（softmax T=1）
  ZS-top3:    只用最相似的 3 个源项目校准
"""
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn

DATA_DIR  = "../Data/GPT2SP Data/Split/"
CACHE_DIR = "../Data/Embeddings/"
DATAS = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement",
         "duracloud", "jirasoftware", "mesos", "moodle", "mule", "mulestudio",
         "springxd", "talenddataquality", "talendesb", "titanium", "usergrid"]
EPOCHS     = 1000
BATCH_SIZE = 32
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
FIBS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

def nearest_fib(x):
    return min(FIBS, key=lambda f: abs(f - x))

def load_sp(name, split):
    return pd.read_csv(f"{DATA_DIR}{name}_{split}.csv")["Storypoint"].values.astype(np.float32)

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)

def run(test_proj, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    src_projs = [d for d in DATAS if d != test_proj]

    # ── 每个源项目的 μ/σ 和 embedding 均值 ──────────────────────────
    proj_stats = {}
    proj_emb_mean = {}
    for p in src_projs:
        sp_all = np.concatenate([load_sp(p, s) for s in ["train","val"]])
        proj_stats[p] = (sp_all.mean(), max(sp_all.std(), 1e-3))
        embs = np.vstack([np.load(f"{CACHE_DIR}mpnet_{p}_{s}.npy") for s in ["train","val"]])
        proj_emb_mean[p] = embs.mean(axis=0)

    # ── 目标项目 embedding 均值（无标签）──────────────────────────────
    X_te = np.vstack([np.load(f"{CACHE_DIR}mpnet_{test_proj}_{s}.npy") for s in ["train","val","test"]])
    y_te = np.concatenate([load_sp(test_proj, s) for s in ["train","val","test"]])
    tgt_emb_mean = X_te.mean(axis=0)

    # ── 组装 z-score 训练数据 ────────────────────────────────────────
    X_list, z_list = [], []
    for p in src_projs:
        mu, sigma = proj_stats[p]
        for s in ["train","val"]:
            emb = np.load(f"{CACHE_DIR}mpnet_{p}_{s}.npy")
            sp  = load_sp(p, s)
            X_list.append(emb)
            z_list.append((sp - mu) / sigma)

    X_tr = np.vstack(X_list)
    y_tr = np.concatenate(z_list).astype(np.float32)

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

    # ── 相似度 ────────────────────────────────────────────────────────
    raw_sims = np.array([cosine_sim(tgt_emb_mean, proj_emb_mean[p]) for p in src_projs])

    # ── 校准策略 1：全局均值/方差（不加权）────────────────────────────
    mu_all  = np.mean([proj_stats[p][0] for p in src_projs])
    sig_all = np.mean([proj_stats[p][1] for p in src_projs])
    sp_global = z_pred * sig_all + mu_all
    mae_global     = sklearn.metrics.mean_absolute_error(y_te, sp_global)
    mae_global_fib = sklearn.metrics.mean_absolute_error(y_te, np.array([nearest_fib(p) for p in sp_global]))

    # ── 校准策略 2：相似度加权（softmax T=1）──────────────────────────
    w = np.exp(raw_sims) / np.exp(raw_sims).sum()
    mu_sim  = sum(w[i] * proj_stats[p][0] for i, p in enumerate(src_projs))
    sig_sim = sum(w[i] * proj_stats[p][1] for i, p in enumerate(src_projs))
    sp_sim = z_pred * sig_sim + mu_sim
    mae_sim     = sklearn.metrics.mean_absolute_error(y_te, sp_sim)
    mae_sim_fib = sklearn.metrics.mean_absolute_error(y_te, np.array([nearest_fib(p) for p in sp_sim]))

    # ── 校准策略 3：Top-3 最相似项目均值──────────────────────────────
    top3_idx = np.argsort(raw_sims)[::-1][:3]
    top3_projs = [src_projs[i] for i in top3_idx]
    mu_top3  = np.mean([proj_stats[p][0] for p in top3_projs])
    sig_top3 = np.mean([proj_stats[p][1] for p in top3_projs])
    sp_top3 = z_pred * sig_top3 + mu_top3
    mae_top3     = sklearn.metrics.mean_absolute_error(y_te, sp_top3)
    mae_top3_fib = sklearn.metrics.mean_absolute_error(y_te, np.array([nearest_fib(p) for p in sp_top3]))

    return {
        "global": (mae_global, mae_global_fib, mu_all, sig_all),
        "sim":    (mae_sim,    mae_sim_fib,    mu_sim, sig_sim),
        "top3":   (mae_top3,   mae_top3_fib,   mu_top3, sig_top3, top3_projs),
    }

BASELINES = {"mesos": 1.914, "moodle": 12.21, "datamanagement": 7.41}
ENG_BEST  = {"mesos": 1.719, "moodle": 12.632, "datamanagement": 7.469}

print("=== Z-score 归一化 + 相似度校准（mpnet + AdamW）===\n")
print(f"{'项目':<18} {'基线':>7} {'工程最佳':>9} {'ZS-global+Fib':>14} {'ZS-sim+Fib':>11} {'ZS-top3+Fib':>12}")

for test_proj in ["mesos", "moodle", "datamanagement"]:
    print(f"\n--- {test_proj} ---")
    res = run(test_proj)

    base = BASELINES[test_proj]
    eng  = ENG_BEST[test_proj]

    g_raw, g_fib, g_mu, g_sig       = res["global"]
    s_raw, s_fib, s_mu, s_sig       = res["sim"]
    t_raw, t_fib, t_mu, t_sig, t3ps = res["top3"]

    print(f"  基线={base:.3f}  工程最佳={eng:.3f}")
    print(f"  ZS-global: 估计μ={g_mu:.1f} σ={g_sig:.1f}  → MAE(raw)={g_raw:.3f}  MAE(+Fib)={g_fib:.3f}")
    print(f"  ZS-sim:    估计μ={s_mu:.1f} σ={s_sig:.1f}  → MAE(raw)={s_raw:.3f}  MAE(+Fib)={s_fib:.3f}")
    print(f"  ZS-top3:   估计μ={t_mu:.1f} σ={t_sig:.1f}  top3={t3ps}  → MAE(raw)={t_raw:.3f}  MAE(+Fib)={t_fib:.3f}")
