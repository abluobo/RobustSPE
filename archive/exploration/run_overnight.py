"""
全夜评估：16 项目 LOPO 完整对比
运行约 6-8 小时，睡前启动

配置 A：工程最佳（mpnet + sqrt + AdamW + Fib）— 3 seed 均值
配置 B：Few-shot K=20, λ=10（z-score + Bayesian 校准）— 10 次重复均值
论文基线：hil-se SBERT-Regression，使用论文报告数字（已验证复现 mesos=1.917≈1.91）
"""
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
import json, time, os

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
EPOCHS     = 1000
BATCH_SIZE = 32
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
FIBS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
LAMBDA = 10
K_FEW  = 20
N_CALIB_REPEAT = 10
N_ENG_SEEDS    = 3
RESULT_FILE = "../results_overnight.json"

def nearest_fib(x):
    return min(FIBS, key=lambda f: abs(f - x))

def load_sp(name, split):
    return pd.read_csv(f"{DATA_DIR}{name}_{split}.csv")["Storypoint"].values.astype(np.float32)

class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)

def train_linear(X_tr, y_tr, X_te, transform="sqrt", weight_decay=1e-3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    if transform == "sqrt":
        y_fit = np.sqrt(np.clip(y_tr, 0, None))
    elif transform == "zscore":
        y_fit = y_tr  # 已经是 z-score
    else:
        y_fit = y_tr

    idx = np.random.permutation(len(X_tr))
    X_tr_t = torch.tensor(X_tr[idx], dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_fit[idx].astype(np.float32), dtype=torch.float32).to(DEVICE)
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

    model = Linear(X_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
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
        preds = model(X_te_t).cpu().numpy()
    return preds

# ── 结果存储 ──────────────────────────────────────────────────────
results = {}
start_time = time.time()

print(f"开始全量评估，设备: {DEVICE}")
print(f"配置A: mpnet+sqrt+AdamW+Fib ({N_ENG_SEEDS} seeds)")
print(f"配置B: z-score+Few-shot K={K_FEW}, λ={LAMBDA} ({N_CALIB_REPEAT} repeats)")
print("="*70)

for proj_idx, test_proj in enumerate(DATAS):
    t0 = time.time()
    print(f"\n[{proj_idx+1:2d}/16] {test_proj}", flush=True)
    src_projs = [d for d in DATAS if d != test_proj]

    # ── 加载数据 ────────────────────────────────────────────────────
    X_tr_mp = np.vstack([np.load(f"{CACHE_DIR}mpnet_{p}_{s}.npy")
                          for p in src_projs for s in ["train","val"]])
    y_tr     = np.concatenate([load_sp(p, s) for p in src_projs for s in ["train","val"]])
    X_te_mp  = np.vstack([np.load(f"{CACHE_DIR}mpnet_{test_proj}_{s}.npy")
                           for s in ["train","val","test"]])
    y_te     = np.concatenate([load_sp(test_proj, s) for s in ["train","val","test"]])

    # ── 配置A：工程最佳（mpnet + sqrt + AdamW + Fib）────────────────
    eng_maes = []
    for seed in range(N_ENG_SEEDS):
        preds = train_linear(X_tr_mp, y_tr, X_te_mp, transform="sqrt",
                             weight_decay=1e-3, seed=seed)
        preds = np.clip(preds, 0, None) ** 2
        preds_fib = np.array([nearest_fib(p) for p in preds])
        eng_maes.append(sklearn.metrics.mean_absolute_error(y_te, preds_fib))
    mae_eng = float(np.mean(eng_maes))
    print(f"  A (工程最佳):  {mae_eng:.3f}  (vs 基线 {PAPER_BASELINE[test_proj]:.2f})", flush=True)

    # ── 配置B：z-score + Few-shot ──────────────────────────────────
    proj_stats = {}
    for p in src_projs:
        sp = np.concatenate([load_sp(p, s) for s in ["train","val"]])
        proj_stats[p] = (float(sp.mean()), float(max(sp.std(), 1e-3)))

    mu_global  = float(np.mean([proj_stats[p][0] for p in src_projs]))
    sig_global = float(np.mean([proj_stats[p][1] for p in src_projs]))

    # z-score 训练数据
    X_list, z_list = [], []
    for p in src_projs:
        mu, sigma = proj_stats[p]
        for s in ["train","val"]:
            emb = np.load(f"{CACHE_DIR}mpnet_{p}_{s}.npy")
            sp  = load_sp(p, s)
            X_list.append(emb)
            z_list.append((sp - mu) / sigma)
    X_tr_z = np.vstack(X_list)
    y_tr_z = np.concatenate(z_list)

    z_pred = train_linear(X_tr_z, y_tr_z, X_te_mp, transform="zscore",
                          weight_decay=1e-3, seed=0)

    # 10 次随机 K=20 校准
    rng = np.random.default_rng(42)
    N = len(y_te)
    few_maes = []
    for _ in range(N_CALIB_REPEAT):
        calib_idx = rng.choice(N, min(K_FEW, N-1), replace=False)
        eval_mask = np.ones(N, dtype=bool); eval_mask[calib_idx] = False
        mu_few  = float(y_te[calib_idx].mean())
        sig_few = float(max(y_te[calib_idx].std(), 1e-3))
        mu_T  = (K_FEW * mu_few  + LAMBDA * mu_global)  / (K_FEW + LAMBDA)
        sig_T = (K_FEW * sig_few + LAMBDA * sig_global) / (K_FEW + LAMBDA)
        sp_pred = z_pred[eval_mask] * sig_T + mu_T
        preds_fib = np.array([nearest_fib(p) for p in sp_pred])
        few_maes.append(sklearn.metrics.mean_absolute_error(y_te[eval_mask], preds_fib))
    mae_few = float(np.mean(few_maes))
    print(f"  B (few-shot K={K_FEW}): {mae_few:.3f}  (vs 基线 {PAPER_BASELINE[test_proj]:.2f})", flush=True)

    results[test_proj] = {
        "paper_baseline": PAPER_BASELINE[test_proj],
        "engineering":    mae_eng,
        "fewshot_k20":    mae_few,
    }
    print(f"  耗时: {time.time()-t0:.0f}s", flush=True)

# ── 汇总 ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("全量结果汇总")
print(f"{'项目':<22} {'论文基线':>10} {'A工程最佳':>10} {'B-Few-K20':>10} {'B-base':>8} {'A-base':>8}")
for proj in DATAS:
    r = results[proj]
    b = r["paper_baseline"]
    e = r["engineering"]
    f = r["fewshot_k20"]
    print(f"{proj:<22} {b:>10.3f} {e:>10.3f} {f:>10.3f} {f-b:>+8.3f} {e-b:>+8.3f}")

avgs = {k: np.mean([results[p][k] for p in DATAS]) for k in ["paper_baseline","engineering","fewshot_k20"]}
print(f"\n{'平均':<22} {avgs['paper_baseline']:>10.3f} {avgs['engineering']:>10.3f} {avgs['fewshot_k20']:>10.3f} "
      f"{avgs['fewshot_k20']-avgs['paper_baseline']:>+8.3f} {avgs['engineering']-avgs['paper_baseline']:>+8.3f}")

print(f"\n总耗时: {(time.time()-start_time)/3600:.1f} 小时")

# 保存结果
with open(RESULT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"结果已保存至: {RESULT_FILE}")
