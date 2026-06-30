"""
K / λ 消融实验
核心优化：每个 test_proj 只训练一次模型，对所有 (K, λ) 组合复用 z_pred
预计耗时 ~3 小时（vs 分开跑需要 10+ 小时）
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

K_VALUES      = [5, 10, 20, 30, 50]
LAMBDA_VALUES = [1, 5, 10, 20, 50]
N_REPEAT = 10
EPOCHS   = 1000
BATCH_SIZE = 32
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
FIBS   = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
RESULT_FILE = "../results_ablation.json"

def nearest_fib(x):
    return min(FIBS, key=lambda f: abs(f - x))

def load_sp(name, split):
    return pd.read_csv(f"{DATA_DIR}{name}_{split}.csv")["Storypoint"].values.astype(np.float32)

class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)

results = {f"K{k}_L{lam}": {} for k in K_VALUES for lam in LAMBDA_VALUES}
start_time = time.time()

print(f"设备: {DEVICE}")
print(f"K 候选: {K_VALUES}")
print(f"λ 候选: {LAMBDA_VALUES}")
print(f"每组合 {N_REPEAT} 次重复")
print("=" * 70)

for proj_idx, test_proj in enumerate(DATAS):
    t0 = time.time()
    print(f"\n[{proj_idx+1:2d}/16] {test_proj}", flush=True)
    src_projs = [d for d in DATAS if d != test_proj]

    # 源项目统计量
    proj_stats = {}
    for p in src_projs:
        sp = np.concatenate([load_sp(p, s) for s in ["train", "val"]])
        proj_stats[p] = (float(sp.mean()), float(max(sp.std(), 1e-3)))

    # 稳健全局参照
    all_src_sp = np.concatenate([
        np.concatenate([load_sp(p, s) for s in ["train", "val"]]) for p in src_projs
    ])
    med_global = float(np.median(all_src_sp))
    mad_global = float(max(np.median(np.abs(all_src_sp - np.median(all_src_sp))) / 0.675, 1e-3))

    # 训练数据
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

    # 训练模型（一次）
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

    # 对所有 (K, λ) 组合复用 z_pred
    rng = np.random.default_rng(42)
    N = len(y_te)
    max_K = max(K_VALUES)

    # 预抽 N_REPEAT 组索引（用最大 K 抽，小 K 取子集）
    all_calib_idx = [rng.choice(N, min(max_K, N - 1), replace=False) for _ in range(N_REPEAT)]

    for K in K_VALUES:
        for LAMBDA in LAMBDA_VALUES:
            maes = []
            for rep_idx in range(N_REPEAT):
                calib_idx = all_calib_idx[rep_idx][:K]
                eval_mask = np.ones(N, dtype=bool); eval_mask[calib_idx] = False

                med_few = float(np.median(y_te[calib_idx]))
                mad_few = float(max(np.median(np.abs(y_te[calib_idx] - med_few)) / 0.675, 1e-3))
                mu_T    = (K * med_few  + LAMBDA * med_global) / (K + LAMBDA)
                sig_T   = (K * mad_few  + LAMBDA * mad_global) / (K + LAMBDA)

                sp_pred   = z_pred[eval_mask] * sig_T + mu_T
                preds_fib = np.array([nearest_fib(p) for p in sp_pred])
                maes.append(sklearn.metrics.mean_absolute_error(y_te[eval_mask], preds_fib))

            key = f"K{K}_L{LAMBDA}"
            results[key][test_proj] = float(np.mean(maes))

    elapsed = time.time() - t0
    # 打印当前项目在各 K（λ=10）下的结果
    row = "  K=[" + ", ".join(f"{results[f'K{k}_L10'][test_proj]:.3f}" for k in K_VALUES) + f"]  耗时{elapsed:.0f}s"
    print(row, flush=True)

# 汇总
print("\n" + "=" * 70)
print("K 消融（λ=10 固定）— 各 K 的平均 MAE（16项目）")
print(f"{'K':>5}", end="")
for k in K_VALUES:
    avg = np.mean(list(results[f"K{k}_L10"].values()))
    print(f"  K={k}:{avg:.3f}", end="")
print()

print("\nλ 消融（K=20 固定）— 各 λ 的平均 MAE（16项目）")
for lam in LAMBDA_VALUES:
    avg = np.mean(list(results[f"K20_L{lam}"].values()))
    print(f"  λ={lam}: {avg:.3f}")

print(f"\n总耗时: {(time.time()-start_time)/3600:.1f} 小时")

with open(RESULT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"结果已保存至: {RESULT_FILE}")
