"""
自适应 SP 取整实验

问题：Fibonacci 取整隐含假设目标项目使用 Fibonacci 刻度，但 moodle(69%)、datamanagement(54%) 大量使用非 Fibonacci 值

三种取整策略对比（mpnet + sqrt + AdamW 为骨干）：
  策略A: Fibonacci {1,2,3,5,8,13,21,34,55,89}
  策略B: 源项目数据驱动 —— 取 15 个源项目中出现过的所有 SP 值作为候选
  策略C: 无取整（连续预测）

在 mesos 和 moodle 上同时验证，确认策略 B 在 moodle 上改善而在 mesos 上不退步
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

def nearest(x, candidates):
    return min(candidates, key=lambda c: abs(c - x))

def load_sp(name, split):
    return pd.read_csv(f"{DATA_DIR}{name}_{split}.csv")["Storypoint"].values.astype(np.float32)

class Linear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)
    def forward(self, x): return self.fc(x).squeeze(-1)

def run(test_proj, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    src_projs = [d for d in DATAS if d != test_proj]

    X_tr = np.vstack([np.load(f"{CACHE_DIR}mpnet_{p}_{s}.npy")
                      for p in src_projs for s in ["train","val"]])
    y_tr = np.concatenate([load_sp(p, s) for p in src_projs for s in ["train","val"]])
    X_te = np.vstack([np.load(f"{CACHE_DIR}mpnet_{test_proj}_{s}.npy") for s in ["train","val","test"]])
    y_te = np.concatenate([load_sp(test_proj, s) for s in ["train","val","test"]])

    # 策略 B 候选集：源项目中出现过的所有 SP 值
    src_candidates = sorted(set(int(v) for v in y_tr))

    y_fit = np.sqrt(np.clip(y_tr, 0, None))
    idx = np.random.permutation(len(X_tr))
    X_tr_t = torch.tensor(X_tr[idx], dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_fit[idx], dtype=torch.float32).to(DEVICE)
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
        preds_raw = model(X_te_t).cpu().numpy()
    preds_cont = np.clip(preds_raw, 0, None) ** 2  # 连续预测

    mae_C = sklearn.metrics.mean_absolute_error(y_te, preds_cont)
    mae_A = sklearn.metrics.mean_absolute_error(y_te, np.array([nearest(p, FIBS) for p in preds_cont]))
    mae_B = sklearn.metrics.mean_absolute_error(y_te, np.array([nearest(p, src_candidates) for p in preds_cont]))

    return mae_C, mae_A, mae_B, src_candidates

print("=== 自适应 SP 取整实验（mpnet + sqrt + AdamW）===\n")
print(f"{'项目':<16} {'策略C(无取整)':>14} {'策略A(Fib)':>12} {'策略B(数据驱动)':>16}  {'B-A':>6}")

for test_proj in ["mesos", "moodle", "datamanagement"]:
    mae_C, mae_A, mae_B, cands = run(test_proj)
    diff = mae_B - mae_A
    print(f"{test_proj:<16} {mae_C:>14.3f} {mae_A:>12.3f} {mae_B:>16.3f}  {diff:>+6.3f}")
    print(f"  源项目候选集大小: {len(cands)} 个值  (Fib=10个)")

print(f"\n论文基线参考: mesos=1.914, moodle=12.21, datamanagement=7.41")
