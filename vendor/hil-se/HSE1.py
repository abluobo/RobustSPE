import pandas as pd
import scipy
import numpy as np
from collections import Counter

filepath = "../Data/HSE1.csv"

def overall_results(filepath):
  df = pd.read_csv(filepath)
  columns = ["I1", "I2", "I3"]
  for i in range(1, 11):
    columns.append("D"+str(i))
    columns.append("DT"+str(i)+"_Page Submit")
  for i in range(1, 11):
    columns.append("C"+str(i))
    columns.append("CT"+str(i)+"_Page Submit")
  columns.append("DD_4")
  columns.append("CD_1")
  df = df[columns]
  df = df.drop(df.index[[0, 1, 2, 7, 8, 9]])

  direct_labels = [1, 13, 2, 8, 5, 2, 8, 5, 3, 5]
  comp_labels = [1, -1, -1, -1, 1, 1, -1, 1, 1, -1]

  df_da = pd.DataFrame(columns=["Professional Exp", "Story Point Exp", "Planning Poker", "Pearson's (Direct)", "Spearman's (Direct)", "MAE (Direct)", "Accuracy (Comp)", "Avg. Time (Direct)", "Avg. Time (Comp)"])

  direct_column_names = ["D"+str(i) for i in range(1, 11)]
  comp_column_names = ["C"+str(i) for i in range(1, 11)]
  pearsons = []
  spearmans = []
  MAEs = []
  accuracies = []
  D_times = []
  C_times = []
  D_conf = df["DD_4"].tolist()
  C_conf = df["CD_1"].tolist()

  for index, row in df.iterrows():
    Ds = []
    Cs = []
    DTs = []
    CTs = []
    for col_name, value in row.items():
      if col_name in direct_column_names:
        Ds.append(int(row[col_name]))
      elif col_name in comp_column_names:
        if type(row[col_name])==float:
          continue
        if "1" in row[col_name]:
          Cs.append(1)
        else:
          Cs.append(-1)
      elif "DT" in col_name:
        DTs.append(float(row[col_name]))
      elif "CT" in col_name:
        CTs.append(float(row[col_name]))
    ps = scipy.stats.pearsonr(Ds, direct_labels)[0]
    sps = scipy.stats.spearmanr(Ds, direct_labels).statistic
    MAE = 0
    acc = 0
    for i in range(len(direct_labels)):
      MAE+=abs(direct_labels[i]-int(Ds[i]))
      if len(Cs)==0:
        continue
      elif comp_labels[i]==int(Cs[i]):
        acc+=1
    MAE/=len(direct_labels)
    acc/=len(comp_labels)
    Dt = sum(DTs)/len(DTs)
    Ct = sum(CTs)/len(CTs)
    pearsons.append(ps)
    spearmans.append(sps)
    MAEs.append(MAE)
    accuracies.append(acc)
    D_times.append(Dt)
    C_times.append(Ct)


  df_da["Professional Exp"] = df["I1"].tolist()
  df_da["Story Point Exp"] = df["I2"].tolist()
  df_da["Planning Poker"] = df["I3"].tolist()
  df_da["Pearson's (Direct)"] = pearsons
  df_da["Spearman's (Direct)"] = spearmans
  df_da["MAE (Direct)"] = MAEs
  df_da["Accuracy (Comp)"] = accuracies
  df_da["Avg. Time (Direct)"] = D_times
  df_da["Avg. Time (Comp)"] = C_times
  df_da["Conf (D)"] = D_conf
  df_da["Conf (C)"] = C_conf

  return df_da

def pairwise_direct_results(filepath):
  df = pd.read_csv(filepath)
  columns = []
  for i in range(1, 11):
    columns.append("D"+str(i))
  df = df[columns]
  df = df.drop(df.index[[0, 1, 2, 7, 8, 9]])
  gt = [1, 13, 2, 8, 5, 2, 8, 5, 3, 5]
  df = df.reset_index(drop=True)
  df_T = df.T
  for i in range(5):
    df_T = df_T.rename(columns={i: "P"+str(i+1)})
  df_T = df_T[["P1", "P2", "P3", "P4", "P5"]]

  print(df_T)
  print(df_T.columns)
  print()

  res = []

  for i in range(5):
    pa = [int(x) for x in df_T["P"+str(i+1)].tolist()]
    for j in range(5):
      if i!=j:
        print("P"+str(i+1), "P"+str(j+1))
        pb = [int(x) for x in df_T["P"+str(j+1)].tolist()]

        ps = scipy.stats.pearsonr(pa, pb)[0]
        sps = scipy.stats.spearmanr(pa, pb).statistic
        MAE = 0
        for k in range(len(gt)):
          MAE+=abs(pa[k]-pb[k])
        MAE/=len(gt)
        print(ps)
        print(sps)
        print(MAE)
    print("P"+str(i+1), "GT")
    ps = scipy.stats.pearsonr(pa, gt)[0]
    sps = scipy.stats.spearmanr(pa, gt).statistic
    MAE = 0
    for k in range(len(gt)):
      MAE+=abs(pa[k]-gt[k])
    MAE/=len(gt)
    print(ps)
    print(sps)
    print(MAE)
    print()

def generate_pairs(x1, x2):
    n = len(x1)
    pairs = {"A":[], "B":[], "agree": []}
    for i in range(n):
        for j in range(i+1, n):
            d1 = d2 = 0
            if x1[i]>x1[j]:
                d1 = 1
            elif x1[i]<x1[j]:
                d1 = -1
            if x2[i]>x2[j]:
                d2 = 1
            elif x2[i]<x2[j]:
                d2 = -1
            if d1!=0 and d2!=0:
                pairs["A"].append(d1)
                pairs["B"].append(d2)
                pairs["agree"].append(d1==d2)
    return pairs

def comp_pairs(x1, x2):
    n = len(x1)
    pairs = {"A":[], "B":[], "agree": []}
    for i in range(n):
        pairs["A"].append(x1[i])
        pairs["B"].append(x2[i])
        pairs["agree"].append(x1[i]==x2[i])
    return pairs

def acc_kappa(pairs):
    n = len(pairs["agree"])
    acc = np.sum(pairs["agree"]) / n
    count_A = Counter(pairs["A"])
    count_B = Counter(pairs["B"])
    pe = n**(-2)*(count_A[1]*count_B[1]+count_A[-1]*count_B[-1]+count_A[0]*count_B[0])
    kappa = 1-(1-acc)/(1-pe)
    return acc, kappa

def pairwise_overall_results(filepath):
  df_c = pd.read_csv(filepath)
  columns = []
  for i in range(1, 11):
    columns.append("C"+str(i))
  comp_column_names = ["C"+str(i) for i in range(1, 11)]
  for col in comp_column_names:
    df_c[col] = np.where(df_c[col]=="Item 1 has a larger story point", 1, -1).astype(int)
  df_c = df_c[columns]
  df_c = df_c.drop(df_c.index[[0, 1, 2, 7, 8]])
  comp_labels = [1, -1, -1, -1, 1, 1, -1, 1, 1, -1]
  df_c = df_c.reset_index(drop=True)
  df_c = df_c.T
  df_c["Ground_Truth"] = comp_labels
  for i in range(6):
    df_c = df_c.rename(columns={i: "P"+str(i)})
  df_c = df_c[["Ground_Truth", "P0", "P1", "P2", "P3", "P4", "P5"]]

  df_d = pd.read_csv(filepath)
  columns = []
  for i in range(1, 11):
    columns.append("D"+str(i))
  df_d = df_d[columns]
  df_d = df_d.drop(df_d.index[[0, 1, 2, 7, 8]])
  direct_labels = [1, 13, 2, 8, 5, 2, 8, 5, 3, 5]
  df_d = df_d.reset_index(drop=True)
  df_d = df_d.T
  df_d["Ground_Truth"] = direct_labels
  for i in range(6):
    df_d = df_d.rename(columns={i: "P"+str(i)})
  df_d = df_d[["Ground_Truth", "P0", "P1", "P2", "P3", "P4", "P5"]]

  raters = ["Ground_Truth", "P0", "P1","P2","P3", "P5"]
  results = []
  for i in range(len(raters)):
      for j in range(i+1, len(raters)):
          pairs_d = generate_pairs(df_d[raters[i]], df_d[raters[j]])
          pairs_c = comp_pairs(df_c[raters[i]], df_c[raters[j]])
          acc_d, kappa_d = acc_kappa(pairs_d)
          acc_c, kappa_c = acc_kappa(pairs_c)
          result = {"Pair": raters[i]+"/"+raters[j], "p_o_D": "%.2f" %acc_d, "kappa_D": "%.2f" %kappa_d, "p_o_C": "%.2f" %acc_c, "kappa_C": "%.2f" %kappa_c}
          results.append(result)

  result_df = pd.DataFrame(results)
  return result_df

df_da = overall_results(filepath)
print(df_da)
pairwise_direct_results(filepath)
df_ov = pairwise_overall_results(filepath)
print(df_ov)