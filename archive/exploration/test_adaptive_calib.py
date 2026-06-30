"""
自适应校准：用 K_v 验证集自动选择 μ+σ vs median+MAD

策略：
  1. 用 K_c 条样本分别计算两种校准参数
  2. 在 K_v 条样本上分别评估 MAE
  3. 选择在验证集上更优的方案用于测试集

预期：
  - mesos: 两者均好，选 median+MAD（更优）
  - moodle: μ+σ 更优，验证集能分辨
  - datamanagement: median+MAD 更优，验证集能分辨
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
LAMBDA = 10
K = 20
K_V = 5
K_C = K - K_V
N_REPEAT = 10

def nearest_fib(x):
    return min(FIBS, key=lambda f: abs(f - x))

fib_arr = lambda arr: np.array([nearest_fib(p) for p in arr])
mae = lambda y, p: sklearn.metrics.mean_absolute_error(y, p)

def load_sp(name, split):
    return pd.read_csv(f"{DATA_DIR}{name}_{split}.csv")["Storypoint"].values.astype(np.float32)

class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)

def train_zscore_model(test_proj, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    src_projs = [d for d in DATAS if d != test_proj]

    proj_stats = {}
    for p in src_projs:
        sp = np.concatenate([load_sp(p, s) for s in ["train","val"]])
        proj_stats[p] = (float(sp.mean()), float(max(sp.std(), 1e-3)))

    mu_global  = float(np.mean([proj_stats[p][0] for p in src_projs]))
    sig_global = float(np.mean([proj_stats[p][1] for p in src_projs]))

    all_sp = np.concatenate([np.concatenate([load_sp(p,s) for s in ["train","val"]]) for p in src_projs])
    med_global = float(np.median(all_sp))
    mad_global = float(np.median(np.abs(all_sp - np.median(all_sp))) / 0.675)

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

    return z_pred, y_te, mu_global, sig_global, med_global, mad_global


TEST_PROJS = ["mesos", "moodle", "datamanagement"]
BASELINES  = {"mesos": 1.914, "moodle": 12.21, "datamanagement": 7.41}
PREV = {
    "μ+σ":      {"mesos": 1.509, "moodle": 11.478, "datamanagement": 9.621},
    "med+MAD":  {"mesos": 1.408, "moodle": 12.016, "datamanagement": 7.343},
}

print("=== 自适应校准实验：K_v 验证集自动选择 μ+σ vs median+MAD ===\n")
print(f"K={K}（K_c={K_C} + K_v={K_V}），λ={LAMBDA}，{N_REPEAT} 次重复\n")

rng = np.random.default_rng(42)

for proj in TEST_PROJS:
    print(f"训练 {proj}...")
    z_pred, y_te, mu_g, sig_g, med_g, mad_g = train_zscore_model(proj)
    N = len(y_te)

    maes_adapt, chosen = [], {"μ+σ": 0, "median+MAD": 0}

    for _ in range(N_REPEAT):
        all_idx = rng.choice(N, K, replace=False)
        c_idx, v_idx = all_idx[:K_C], all_idx[K_C:]
        mask = np.ones(N, dtype=bool); mask[all_idx] = False

        # 方案1：μ+σ 校准参数（K_c 条）
        mu_f   = float(y_te[c_idx].mean())
        sig_f  = float(max(y_te[c_idx].std(), 1e-3))
        mu_T1  = (K_C * mu_f  + LAMBDA * mu_g)  / (K_C + LAMBDA)
        sig_T1 = (K_C * sig_f + LAMBDA * sig_g) / (K_C + LAMBDA)

        # 方案2：median+MAD 校准参数（K_c 条）
        med_f  = float(np.median(y_te[c_idx]))
        mad_f  = float(max(np.median(np.abs(y_te[c_idx] - med_f)) / 0.675, 1e-3))
        mu_T2  = (K_C * med_f + LAMBDA * med_g) / (K_C + LAMBDA)
        sig_T2 = (K_C * mad_f + LAMBDA * mad_g) / (K_C + LAMBDA)

        # 在 K_v 上比较
        pred1_v = fib_arr(z_pred[v_idx] * sig_T1 + mu_T1)
        pred2_v = fib_arr(z_pred[v_idx] * sig_T2 + mu_T2)
        mae1_v  = mae(y_te[v_idx], pred1_v)
        mae2_v  = mae(y_te[v_idx], pred2_v)

        if mae1_v <= mae2_v:
            pred = fib_arr(z_pred[mask] * sig_T1 + mu_T1); chosen["μ+σ"] += 1
        else:
            pred = fib_arr(z_pred[mask] * sig_T2 + mu_T2); chosen["median+MAD"] += 1

        maes_adapt.append(mae(y_te[mask], pred))

    avg = np.mean(maes_adapt)
    b = BASELINES[proj]
    print(f"  基线={b:.3f}")
    print(f"  纯 μ+σ={PREV['μ+σ'][proj]:.3f}   纯 med+MAD={PREV['med+MAD'][proj]:.3f}")
    print(f"  自适应={avg:.3f}  ({avg-b:+.3f} vs 基线)  "
          f"μ+σ选{chosen['μ+σ']}次, med+MAD选{chosen['median+MAD']}次\n")
