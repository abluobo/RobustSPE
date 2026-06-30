"""
稳健校准（Robust Calibration）修复 datamanagement 重尾问题

问题：datamanagement σ=16.60，均值被极少数大值拉高
  - μ=9.57，但 69% 的 issue SP≤6，中位数约 3-5
  - K=20 的 std 估计极不稳定（2.6~29.6）

稳健方案：
  - 位置估计：中位数（而非均值）→ 对极端值免疫
  - 尺度估计：MAD/0.675（而非 std）→ 对极端值免疫

对比三种校准方式：
  A. 原始 μ+σ（K=20, λ=10）
  B. 稳健 median+MAD（K=20, λ=10）
  C. 稳健 + auto-fallback（K_c=15, K_v=5）
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

    # 稳健全局统计量
    all_sp = np.concatenate([np.concatenate([load_sp(p,s) for s in ["train","val"]]) for p in src_projs])
    mu_global    = float(np.mean([proj_stats[p][0] for p in src_projs]))
    sig_global   = float(np.mean([proj_stats[p][1] for p in src_projs]))
    med_global   = float(np.median(all_sp))
    mad_global   = float(np.median(np.abs(all_sp - np.median(all_sp))) / 0.675)

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
ORIG_K20   = {"mesos": 1.518, "moodle": 11.543, "datamanagement": 9.573}
AUTOFB     = {"mesos": 1.587, "moodle": 11.557, "datamanagement": 7.872}

print("=== 稳健校准实验 ===\n")
print(f"K={K}, λ={LAMBDA}, {N_REPEAT} 次重复\n")

rng = np.random.default_rng(42)

for proj in TEST_PROJS:
    print(f"训练 {proj}...")
    z_pred, y_te, mu_g, sig_g, med_g, mad_g = train_zscore_model(proj)
    N = len(y_te)

    # 打印全局稳健统计量
    sp_te = y_te  # 目标项目真实 SP
    med_te = float(np.median(sp_te))
    mad_te = float(np.median(np.abs(sp_te - med_te)) / 0.675)
    print(f"  目标项目统计: μ={sp_te.mean():.2f}, σ={sp_te.std():.2f}, 中位={med_te:.1f}, MAD/0.675={mad_te:.2f}")
    print(f"  全局参照: μ_g={mu_g:.2f}, σ_g={sig_g:.2f}, 中位_g={med_g:.1f}, MAD_g={mad_g:.2f}")

    # === 方案 A：原始 μ+σ ===
    maes_A = []
    rng_A = np.random.default_rng(42)
    for _ in range(N_REPEAT):
        idx = rng_A.choice(N, K, replace=False)
        mask = np.ones(N, dtype=bool); mask[idx] = False
        mu_f   = float(y_te[idx].mean())
        sig_f  = float(max(y_te[idx].std(), 1e-3))
        mu_T   = (K*mu_f  + LAMBDA*mu_g)  / (K+LAMBDA)
        sig_T  = (K*sig_f + LAMBDA*sig_g) / (K+LAMBDA)
        pred = fib_arr(z_pred[mask] * sig_T + mu_T)
        maes_A.append(sklearn.metrics.mean_absolute_error(y_te[mask], pred))

    # === 方案 B：稳健 median+MAD ===
    maes_B = []
    rng_B = np.random.default_rng(42)
    for _ in range(N_REPEAT):
        idx = rng_B.choice(N, K, replace=False)
        mask = np.ones(N, dtype=bool); mask[idx] = False
        med_f = float(np.median(y_te[idx]))
        mad_f = float(max(np.median(np.abs(y_te[idx] - med_f)) / 0.675, 1e-3))
        mu_T  = (K*med_f + LAMBDA*med_g) / (K+LAMBDA)   # 用中位数代替均值
        sig_T = (K*mad_f + LAMBDA*mad_g) / (K+LAMBDA)   # 用MAD代替std
        pred = fib_arr(z_pred[mask] * sig_T + mu_T)
        maes_B.append(sklearn.metrics.mean_absolute_error(y_te[mask], pred))

    # === 方案 C：稳健 + auto-fallback ===
    maes_C, used_C = [], {"calibrated": 0, "fallback": 0}
    rng_C = np.random.default_rng(42)
    for _ in range(N_REPEAT):
        all_idx = rng_C.choice(N, K, replace=False)
        c_idx, v_idx = all_idx[:K_C], all_idx[K_C:]
        mask = np.ones(N, dtype=bool); mask[all_idx] = False

        # 稳健校准参数（用 K_C 条）
        med_f = float(np.median(y_te[c_idx]))
        mad_f = float(max(np.median(np.abs(y_te[c_idx] - med_f)) / 0.675, 1e-3))
        mu_T  = (K_C*med_f + LAMBDA*med_g) / (K_C+LAMBDA)
        sig_T = (K_C*mad_f + LAMBDA*mad_g) / (K_C+LAMBDA)

        # 在 K_V 验证集上比较
        mae_calib_v  = sklearn.metrics.mean_absolute_error(y_te[v_idx], fib_arr(z_pred[v_idx]*sig_T+mu_T))
        mae_global_v = sklearn.metrics.mean_absolute_error(y_te[v_idx], fib_arr(z_pred[v_idx]*mad_g+med_g))

        if mae_calib_v <= mae_global_v:
            pred = fib_arr(z_pred[mask]*sig_T+mu_T); used_C["calibrated"]+=1
        else:
            pred = fib_arr(z_pred[mask]*mad_g+med_g); used_C["fallback"]+=1

        maes_C.append(sklearn.metrics.mean_absolute_error(y_te[mask], pred))

    import sklearn.metrics
    b = BASELINES[proj]
    print(f"  基线={b:.3f}  原K20={ORIG_K20[proj]:.3f}  auto-fb={AUTOFB[proj]:.3f}")
    print(f"  A (μ+σ):    {np.mean(maes_A):.3f}  ({np.mean(maes_A)-b:+.3f} vs 基线)")
    print(f"  B (med+MAD):{np.mean(maes_B):.3f}  ({np.mean(maes_B)-b:+.3f} vs 基线)")
    print(f"  C (稳健+fb):{np.mean(maes_C):.3f}  ({np.mean(maes_C)-b:+.3f} vs 基线)  "
          f"校准{used_C['calibrated']}/回退{used_C['fallback']}\n")
