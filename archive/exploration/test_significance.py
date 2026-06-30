"""
显著性检验 + 失败项目特征分析
"""
import json
import numpy as np
from scipy import stats

with open("../results_robust_full.json") as f:
    results = json.load(f)

projs = list(results.keys())
baseline = np.array([results[p]["paper_baseline"] for p in projs])
musig    = np.array([results[p]["musig_k20"]       for p in projs])
robust   = np.array([results[p]["robust_k20"]      for p in projs])

print("=" * 60)
print("Wilcoxon Signed-Rank Test（单侧，H1: med+MAD < baseline）")
print("=" * 60)

# med+MAD vs paper baseline
stat, p = stats.wilcoxon(robust, baseline, alternative="less")
print(f"\nmed+MAD vs 论文基线:")
print(f"  W={stat:.1f}, p={p:.4f} {'*** p<0.001' if p<0.001 else ('** p<0.01' if p<0.01 else ('* p<0.05' if p<0.05 else 'n.s.'))}")

# μ+σ vs paper baseline
stat2, p2 = stats.wilcoxon(musig, baseline, alternative="less")
print(f"\nμ+σ vs 论文基线:")
print(f"  W={stat2:.1f}, p={p2:.4f} {'*** p<0.001' if p2<0.001 else ('** p<0.01' if p2<0.01 else ('* p<0.05' if p2<0.05 else 'n.s.'))}")

# med+MAD vs μ+σ（双侧）
stat3, p3 = stats.wilcoxon(robust, musig)
print(f"\nmed+MAD vs μ+σ（双侧）:")
print(f"  W={stat3:.1f}, p={p3:.4f} {'*** p<0.001' if p3<0.001 else ('** p<0.01' if p3<0.01 else ('* p<0.05' if p3<0.05 else 'n.s.'))}")

# Effect size (r = Z / sqrt(N))
from scipy.stats import wilcoxon as wil
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = wil(robust, baseline, alternative="less", method="exact")
n = len(projs)
print(f"\n效果量（r = Z/√N，vs 基线）: 需用近似Z值")
# 近似 Z from p
from scipy.stats import norm
z = norm.ppf(p)
r = abs(z) / np.sqrt(n)
print(f"  近似 r = {r:.3f}（>0.3 中等，>0.5 大）")

print("\n" + "=" * 60)
print("失败项目分析（med+MAD > 基线）")
print("=" * 60)

import pandas as pd
import os

DATA_DIR = "../Data/GPT2SP Data/Split/"

fail_projs  = [p for p in projs if results[p]["robust_k20"] > results[p]["paper_baseline"]]
win_projs   = [p for p in projs if results[p]["robust_k20"] <= results[p]["paper_baseline"]]

print(f"\n失败项目（{len(fail_projs)}个）: {fail_projs}")
print(f"成功项目（{len(win_projs)}个）: {win_projs}")

print(f"\n{'项目':<22} {'均值':>7} {'中位数':>7} {'std':>7} {'MAD':>7} {'偏斜':>7} {'N_train':>8} {'退步幅度':>9}")
stats_data = {}
for p in projs:
    sp = []
    for s in ["train", "val", "test"]:
        df = pd.read_csv(f"{DATA_DIR}{p}_{s}.csv")
        sp.extend(df["Storypoint"].tolist())
    sp = np.array(sp, dtype=float)
    mu  = sp.mean()
    med = np.median(sp)
    std = sp.std()
    mad = np.median(np.abs(sp - med)) / 0.675
    skew = stats.skew(sp)
    n_train = len(pd.read_csv(f"{DATA_DIR}{p}_train.csv"))
    delta = results[p]["robust_k20"] - results[p]["paper_baseline"]
    stats_data[p] = {"mu": mu, "median": med, "std": std, "mad": mad, "skew": skew, "n_train": n_train}
    flag = "❌" if p in fail_projs else "✅"
    print(f"{p:<22} {mu:>7.2f} {med:>7.2f} {std:>7.2f} {mad:>7.2f} {skew:>7.2f} {n_train:>8d} {delta:>+9.3f} {flag}")

print("\n失败组 vs 成功组 均值对比:")
for key, label in [("skew","偏斜度"), ("std","std"), ("mad","MAD"), ("n_train","训练集大小"), ("mu","均值SP")]:
    fail_vals = [stats_data[p][key] for p in fail_projs]
    win_vals  = [stats_data[p][key] for p in win_projs]
    print(f"  {label:10s}: 失败组={np.mean(fail_vals):.2f}, 成功组={np.mean(win_vals):.2f}")
