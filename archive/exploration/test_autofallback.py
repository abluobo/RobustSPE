"""
Auto-fallback 修复 datamanagement 失败问题

核心思路：
  K 个样本中，K_c 用于校准，K_v 用于验证
  如果校准后在 K_v 上比全局校准差 → 自动回退到全局校准
  这样对于 σ 极大的项目（datamanagement），不稳定的校准会被自动拦截

对比：
  原始 K=20 λ=10（无 fallback）：datamanagement 9.573，mesos 1.518，moodle 11.543
  K=20 λ=10 + auto-fallback：预期 datamanagement 恢复到 ~7.4，其他项目不变
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
K_V = 5       # 用于验证的样本数
K_C = K - K_V # 用于校准的样本数
N_REPEAT = 10

def nearest_fib(x):
    return min(FIBS, key=lambda f: abs(f - x))

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

    return z_pred, y_te, mu_global, sig_global

def eval_with_fallback(z_pred, y_te, mu_g, sig_g, rng):
    N = len(y_te)
    all_idx = rng.choice(N, K, replace=False)
    calib_idx = all_idx[:K_C]   # 15 条用于校准
    valid_idx = all_idx[K_C:]   # 5 条用于验证
    eval_mask = np.ones(N, dtype=bool)
    eval_mask[all_idx] = False   # 评估剩余样本

    # 校准参数
    mu_few  = float(y_te[calib_idx].mean())
    sig_few = float(max(y_te[calib_idx].std(), 1e-3))
    mu_T    = (K_C * mu_few  + LAMBDA * mu_g) / (K_C + LAMBDA)
    sig_T   = (K_C * sig_few + LAMBDA * sig_g) / (K_C + LAMBDA)

    # 在验证集上比较校准 vs 全局
    fib = lambda arr: np.array([nearest_fib(p) for p in arr])
    sp_calib_val  = fib(z_pred[valid_idx] * sig_T + mu_T)
    sp_global_val = fib(z_pred[valid_idx] * sig_g + mu_g)
    mae_calib_val  = sklearn.metrics.mean_absolute_error(y_te[valid_idx], sp_calib_val)
    mae_global_val = sklearn.metrics.mean_absolute_error(y_te[valid_idx], sp_global_val)

    # 自动选择：校准更好则用校准，否则回退全局
    if mae_calib_val <= mae_global_val:
        sp_pred = fib(z_pred[eval_mask] * sig_T + mu_T)
        used = "calibrated"
    else:
        sp_pred = fib(z_pred[eval_mask] * sig_g + mu_g)
        used = "fallback"

    mae = sklearn.metrics.mean_absolute_error(y_te[eval_mask], sp_pred)
    return mae, used

TEST_PROJS = ["mesos", "moodle", "datamanagement"]
BASELINES  = {"mesos": 1.914, "moodle": 12.21, "datamanagement": 7.41}
ORIG_K20   = {"mesos": 1.518, "moodle": 11.543, "datamanagement": 9.573}

print("=== Auto-fallback 修复实验 ===\n")
print(f"K={K}（K_c={K_C} 校准 + K_v={K_V} 验证），λ={LAMBDA}，{N_REPEAT} 次重复\n")

rng = np.random.default_rng(42)

for proj in TEST_PROJS:
    print(f"训练 {proj}...")
    z_pred, y_te, mu_g, sig_g = train_zscore_model(proj)

    maes, used_counts = [], {"calibrated": 0, "fallback": 0}
    for _ in range(N_REPEAT):
        mae, used = eval_with_fallback(z_pred, y_te, mu_g, sig_g, rng)
        maes.append(mae)
        used_counts[used] += 1

    avg = np.mean(maes)
    print(f"  基线={BASELINES[proj]:.3f}  原K20={ORIG_K20[proj]:.3f}  "
          f"auto-fallback={avg:.3f}  "
          f"(校准{used_counts['calibrated']}次/回退{used_counts['fallback']}次)")
    print(f"  vs 基线: {avg-BASELINES[proj]:+.3f}  vs 原K20: {avg-ORIG_K20[proj]:+.3f}\n")
