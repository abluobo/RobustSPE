"""
Few-shot + Bayesian 混合校准

核心修复：防止小 K 时 μ/σ 估计野性发散
  μ_T = (K * μ_few + λ * μ_global) / (K + λ)
  σ_T = (K * σ_few + λ * σ_global) / (K + λ)

λ=0  退化为纯 few-shot（原始版本，datamanagement 崩溃）
λ=10 小样本时向全局收缩，大样本时信任实测
λ=50 强先验，需要更多样本才能覆盖全局估计

同时测试三种 λ，找最优超参数
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
K_LIST     = [0, 5, 10, 20, 50]
LAMBDA_LIST = [10, 20, 50]
N_REPEAT   = 10

def nearest_fib(x):
    return min(FIBS, key=lambda f: abs(f - x))

def load_sp(name, split):
    return pd.read_csv(f"{DATA_DIR}{name}_{split}.csv")["Storypoint"].values.astype(np.float32)

class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)

def train_model(test_proj, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    src_projs = [d for d in DATAS if d != test_proj]

    proj_stats = {}
    for p in src_projs:
        sp = np.concatenate([load_sp(p, s) for s in ["train","val"]])
        proj_stats[p] = (float(sp.mean()), float(max(sp.std(), 1e-3)))

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
    X_te = np.vstack([np.load(f"{CACHE_DIR}mpnet_{test_proj}_{s}.npy") for s in ["train","val","test"]])
    y_te = np.concatenate([load_sp(test_proj, s) for s in ["train","val","test"]])

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

    mu_global  = float(np.mean([proj_stats[p][0] for p in src_projs]))
    sig_global = float(np.mean([proj_stats[p][1] for p in src_projs]))
    return z_pred, y_te, mu_global, sig_global

def eval_bayes(z_pred, y_te, mu_g, sig_g, K, lam, rng):
    N = len(y_te)
    if K == 0:
        sp_pred = z_pred * sig_g + mu_g
    else:
        calib_idx = rng.choice(N, min(K, N-1), replace=False)
        eval_mask = np.ones(N, dtype=bool); eval_mask[calib_idx] = False
        mu_few  = float(y_te[calib_idx].mean())
        sig_few = float(max(y_te[calib_idx].std(), 1e-3))
        # Bayesian 混合
        mu_T  = (K * mu_few  + lam * mu_g)  / (K + lam)
        sig_T = (K * sig_few + lam * sig_g) / (K + lam)
        sp_pred = z_pred[eval_mask] * sig_T + mu_T
        y_te = y_te[eval_mask]

    preds_fib = np.array([nearest_fib(p) for p in sp_pred])
    return sklearn.metrics.mean_absolute_error(y_te, preds_fib)

BASELINES = {"mesos": 1.914, "moodle": 12.21, "datamanagement": 7.41}
TEST_PROJS = ["mesos", "moodle", "datamanagement"]

print("=== Few-shot + Bayesian 混合校准 ===\n")

# 训练三个项目模型
models = {}
for proj in TEST_PROJS:
    print(f"训练 {proj}...")
    models[proj] = train_model(proj)

rng = np.random.default_rng(42)

for lam in LAMBDA_LIST:
    print(f"\n{'='*60}")
    print(f"λ = {lam}（先验强度：需 >{lam} 条样本才主要依赖实测）")
    print(f"{'K':>5}", end="")
    for proj in TEST_PROJS:
        print(f"  {proj:>18}", end="")
    print()
    print(f"{'基线':>5}", end="")
    for proj in TEST_PROJS:
        print(f"  {BASELINES[proj]:>18.3f}", end="")
    print()

    for K in K_LIST:
        print(f"{K:>5}", end="")
        for proj in TEST_PROJS:
            z_pred, y_te, mu_g, sig_g = models[proj]
            maes = [eval_bayes(z_pred.copy(), y_te.copy(), mu_g, sig_g, K, lam, rng)
                    for _ in range(N_REPEAT)]
            avg = float(np.mean(maes))
            diff = avg - BASELINES[proj]
            print(f"  {avg:>8.3f} ({diff:>+6.3f})", end="")
        print()

print("\n括号内为 vs 论文基线的差值（负=改善）")
