import numpy as np
import pandas as pd
from collections import Counter
import scipy
import math

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

def gen_specific_pairs(x, pairs):
  y = []
  # print("len:", len(x))
  for p in pairs:
    x1 = p[0]-2
    x2 = p[1]-2
    # print(x1, x2)
    a = x[x1]
    b = x[x2]
    # print(a, b)
    if a>b:
      y.append(1)
    elif a<b:
      y.append(-1)
    else:
      y.append(0)
  # print()
  return y

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

group_nums = [[1, 2, 3, 4, 5],
              [1, 2, 3, 4],
              [1, 2, 3, 4, 5],
              [1, 2, 3, 4],
              [1, 2, 3, 4],
              [1, 2, 3, 4, 5]]
group_types = [["C", "D", "C", "D", "C"],
               ["C", "C", "D", "D"],
               ["D", "D", "C", "C", "C"],
               ["D", "D", "C", "C"],
               ["D", "D", "C", "C"],
               ["C", "C", "D", "D", "C"]]

def getResults(df1, df2, P1, P2, comp_type):
  ps = None
  sps = None
  df1.dropna(axis = 0, how = 'all', inplace = True)
  df2.dropna(axis = 0, how = 'all', inplace = True)
  df1.columns = [x.upper() for x in df1.columns]
  df2.columns = [x.upper() for x in df2.columns]
  if comp_type=="Direct":
    l1 = df1["STORY POINT"].tolist()
    l2 = df2["STORY POINT"].tolist()
    l1 = list(filter(lambda x: x is not None and pd.notnull(x), l1))
    l2 = list(filter(lambda x: x is not None and pd.notnull(x), l2))
    min_len = min(len(l1), len(l2))
    l1 = l1[:min_len]
    l2 = l2[:min_len]
    ps = scipy.stats.pearsonr(l1, l2)[0]
    sps = scipy.stats.spearmanr(l1, l2).statistic
    pairs = generate_pairs(l1, l2)
    p_o, kappa = acc_kappa(pairs)
  else:
    df1.columns = df1.columns.str.replace('LABELS', 'LABEL')
    df2.columns = df2.columns.str.replace('LABELS', 'LABEL')
    l1 = df1["LABEL"].tolist()
    l2 = df2["LABEL"].tolist()
    l1 = [x for x in l1 if not math.isnan(x)]
    l2 = [x for x in l2 if not math.isnan(x)]
    min_len = min(len(l1), len(l2))
    l1 = l1[:min_len]
    l2 = l2[:min_len]
    pairs = comp_pairs(l1, l2)
    p_o, kappa = acc_kappa(pairs)
  result = {"P1": P1, "P2": P2,
            "Pearsons": ps, "Spearmans": sps,
            "p_o": p_o, "Kappa": kappa}
  return result


# Group 1
folder = "../Data/HSE-2/1/"
df1 = pd.read_csv(folder+"1_1_comp.csv")
df2 = pd.read_csv(folder+"1_2_direct.csv")
df3 = pd.read_csv(folder+"1_3_comp.csv")
df4 = pd.read_csv(folder+"1_4_direct.csv")
df5 = pd.read_csv(folder+"1_5_comp.csv")

dfd = pd.read_csv(folder+"1_direct_final.csv")
dfc = pd.read_csv(folder+"1_comp_final.csv")

results_1d = []
results_1c = []

results_1d.append(getResults(df2, df4, "2", "4", "Direct"))
results_1d.append(getResults(df2, dfd, "2", "Group", "Direct"))
results_1d.append(getResults(df4, dfd, "4", "Group", "Direct"))

results_1c.append(getResults(df1, df3, "1", "3", "Comp"))
results_1c.append(getResults(df1, df5, "1", "5", "Comp"))
results_1c.append(getResults(df1, dfc, "1", "Group", "Comp"))
results_1c.append(getResults(df3, df5, "3", "5", "Comp"))
results_1c.append(getResults(df3, dfc, "3", "Group", "Comp"))
results_1c.append(getResults(df5, dfc, "5", "Group", "Comp"))

df_g_1d = pd.DataFrame(results_1d)
print("Group-1, Direct")
print(df_g_1d)
print()

df_g_1c = pd.DataFrame(results_1c)
df_g_1c = df_g_1c.dropna(axis=1,how='all')
print("Group-1, Comparative")
print(df_g_1c)
print()

results_q = []
results_q.append({"P": "1", "Type": "Comp", "Time": df1["TOTAL TIME"].tolist()[-1]/len(df1.index)})
results_q.append({"P": "2", "Type": "Direct", "Time": sum(df2["TOTAL TIME - SECONDS"].tolist())/len(df2["TOTAL TIME - SECONDS"].tolist())})
results_q.append({"P": "3", "Type": "Comp", "Time": float(df3["TOTAL TIME"].tolist()[-1])/(len(df3.index)-1)})
temp = df4["TOTAL TIME"].tolist()
temp = [float(x[3:]) for x in temp]
results_q.append({"P": "4", "Type": "Direct", "Time": sum(temp)/len(temp)})
results_q.append({"P": "5", "Type": "Comp", "Time": df5["TOTAL TIME"].tolist()[-1]/(len(df5.index)-1)})
results_q.append({"P": "Group", "Type": "Comp", "Time": dfc["TOTAL TIME"].tolist()[-1]/len(dfc.index)})
results_q.append({"P": "Group", "Type": "Direct", "Time": dfd["TOTAL TIME"].tolist()[-1]/(len(dfc.index)-1)})
df_g_1q = pd.DataFrame(results_q)
print("Group-1, times")
print(df_g_1q)
print()

##########
# Group 2
##########

folder = "../Data/HSE-2/2/"
df1 = pd.read_csv(folder+"2_1_comp.csv")
df2 = pd.read_csv(folder+"2_2_comp.csv")
df3 = pd.read_csv(folder+"2_3_direct.csv")
df4 = pd.read_csv(folder+"2_4_direct.csv")

dfd = pd.read_csv(folder+"2_direct_final.csv")
dfc = pd.read_csv(folder+"2_comp_final.csv")

results_1d = []
results_1c = []

results_1d.append(getResults(df3, df4, "3", "4", "Direct"))
results_1d.append(getResults(df3, dfd, "3", "Group", "Direct"))
results_1d.append(getResults(df4, dfd, "4", "Group", "Direct"))

results_1c.append(getResults(df1, df2, "1", "2", "Comp"))
results_1c.append(getResults(df1, dfc, "1", "Group", "Comp"))
results_1c.append(getResults(df2, dfc, "2", "Group", "Comp"))

df_g_1d = pd.DataFrame(results_1d)
print("Group-2, Direct")
print(df_g_1d)
print()

df_g_1c = pd.DataFrame(results_1c)
df_g_1c = df_g_1c.dropna(axis=1,how='all')
print("Group-2, Comparative")
print(df_g_1c)
print()

results_q = []
results_q.append({"P": "1", "Type": "Comp", "Time": 308.3/len(df1.index)})
results_q.append({"P": "2", "Type": "Comp", "Time": 47.36/len(df2["TOTAL TIME"].tolist())-1})
results_q.append({"P": "3", "Type": "Direct", "Time": 109/(len(df3.index)-1)})
results_q.append({"P": "4", "Type": "Direct", "Time": 121.4/len(df4.index)})
results_q.append({"P": "Group", "Type": "Comp", "Time": 170.4/(len(dfc.index)-1)})
results_q.append({"P": "Group", "Type": "Direct", "Time": 108.9/(len(dfc.index))})

df_g_1q = pd.DataFrame(results_q)
print("Group-2, times")
print(df_g_1q)
print()


##########
# Group 3
##########

folder = "../Data/HSE-2/3/"
df1 = pd.read_csv(folder+"3_1_direct.csv")
df2 = pd.read_csv(folder+"3_2_direct.csv")
df3 = pd.read_csv(folder+"3_3_comp.csv")
df4 = pd.read_csv(folder+"3_4_comp.csv")
df5 = pd.read_csv(folder+"3_5_comp.csv")

dfd = pd.read_csv(folder+"3_direct_final.csv")
dfc = pd.read_csv(folder+"3_comp_final.csv")

results_1d = []
results_1c = []

results_1d.append(getResults(df1, df2, "1", "2", "Direct"))
results_1d.append(getResults(df1, dfd, "1", "Group", "Direct"))
results_1d.append(getResults(df2, dfd, "2", "Group", "Direct"))

results_1c.append(getResults(df3, df4, "3", "4", "Comp"))
results_1c.append(getResults(df3, df5, "3", "5", "Comp"))
results_1c.append(getResults(df4, df5, "4", "5", "Comp"))
results_1c.append(getResults(df3, dfc, "3", "Group", "Comp"))
results_1c.append(getResults(df4, dfc, "4", "Group", "Comp"))
results_1c.append(getResults(df5, dfc, "5", "Group", "Comp"))

df_g_1d = pd.DataFrame(results_1d)
print("Group-3, Direct")
print(df_g_1d)
print()

df_g_1c = pd.DataFrame(results_1c)
df_g_1c = df_g_1c.dropna(axis=1,how='all')
print("Group-3, Comparative")
print(df_g_1c)
print()

results_q = []
results_q.append({"P": "1", "Type": "Direct", "Time": 300/len(df1.index)})
results_q.append({"P": "2", "Type": "Direct", "Time": 225/len(df2.index)})
results_q.append({"P": "3", "Type": "Comp", "Time": 109.27/(len(df3.index))})
results_q.append({"P": "4", "Type": "Comp", "Time": 190/(len(df4.index)-1)})
results_q.append({"P": "5", "Type": "Comp", "Time": 159/(len(df3.index))})
results_q.append({"P": "Group", "Type": "Comp", "Time": 450/(len(dfc.index))})
results_q.append({"P": "Group", "Type": "Direct", "Time": 710/(len(dfc.index))})

df_g_1q = pd.DataFrame(results_q)
print("Group-3, times")
print(df_g_1q)
print()


##########
# Group 4
##########

folder = "../Data/HSE-2/4/"
df1 = pd.read_csv(folder+"4_1_direct.csv")
df2 = pd.read_csv(folder+"4_2_direct.csv")
df3 = pd.read_csv(folder+"4_3_comp.csv")
df4 = pd.read_csv(folder+"4_4_comp.csv")

dfd = pd.read_csv(folder+"4_direct_final.csv")
dfc = pd.read_csv(folder+"4_comp_final.csv")

results_1d = []
results_1c = []

results_1d.append(getResults(df1, df2, "1", "2", "Direct"))
results_1d.append(getResults(df1, dfd, "1", "Group", "Direct"))
results_1d.append(getResults(df2, dfd, "2", "Group", "Direct"))

results_1c.append(getResults(df3, df4, "3", "4", "Comp"))
results_1c.append(getResults(df3, dfc, "3", "Group", "Comp"))
results_1c.append(getResults(df4, dfc, "4", "Group", "Comp"))

df_g_1d = pd.DataFrame(results_1d)
print("Group-4, Direct")
print(df_g_1d)
print()

df_g_1c = pd.DataFrame(results_1c)
df_g_1c = df_g_1c.dropna(axis=1,how='all')
print("Group-4, Comparative")
print(df_g_1c)
print()

results_q = []
results_q.append({"P": "1", "Type": "Direct", "Time": 128.19/len(df1.index)})
results_q.append({"P": "2", "Type": "Direct", "Time": 112/len(df2.index)})
results_q.append({"P": "3", "Type": "Comp", "Time": 99.14/(len(df3.index))})
results_q.append({"P": "4", "Type": "Comp", "Time": 68/(len(df4.index))})
results_q.append({"P": "Group", "Type": "Comp", "Time": 152/(len(dfc.index))})
results_q.append({"P": "Group", "Type": "Direct", "Time": 258/(len(dfc.index))})

df_g_1q = pd.DataFrame(results_q)
print("Group-4, times")
print(df_g_1q)
print()


##########
# Group 5
##########

folder = "../Data/HSE-2/5/"
df1 = pd.read_csv(folder+"5_1_direct.csv")
df2 = pd.read_csv(folder+"5_2_direct.csv")
df3 = pd.read_csv(folder+"5_3_comp.csv")
df4 = pd.read_csv(folder+"5_4_comp.csv")

dfd = pd.read_csv(folder+"5_direct_final.csv")
dfc = pd.read_csv(folder+"5_comp_final.csv")

results_1d = []
results_1c = []

results_1d.append(getResults(df1, df2, "1", "2", "Direct"))
results_1d.append(getResults(df1, dfd, "1", "Group", "Direct"))
results_1d.append(getResults(df2, dfd, "2", "Group", "Direct"))

results_1c.append(getResults(df3, df4, "3", "4", "Comp"))
results_1c.append(getResults(df3, dfc, "3", "Group", "Comp"))
results_1c.append(getResults(df4, dfc, "4", "Group", "Comp"))

df_g_1d = pd.DataFrame(results_1d)
print("Group-5, Direct")
print(df_g_1d)
print()

df_g_1c = pd.DataFrame(results_1c)
df_g_1c = df_g_1c.dropna(axis=1,how='all')
print("Group-5, Comparative")
print(df_g_1c)
print()

results_q = []
results_q.append({"P": "1", "Type": "Direct", "Time": 183/len(df1.index)})
results_q.append({"P": "2", "Type": "Direct", "Time": 86.9/len(df2.index)})
results_q.append({"P": "3", "Type": "Comp", "Time": 184/(len(df3.index))})
results_q.append({"P": "4", "Type": "Comp", "Time": 157/(len(df4.index))})
results_q.append({"P": "Group", "Type": "Comp", "Time": 242/(len(dfc.index))})
results_q.append({"P": "Group", "Type": "Direct", "Time": 464/(len(dfc.index))})

df_g_1q = pd.DataFrame(results_q)
print("Group-5, times")
print(df_g_1q)
print()

##########
# Group 6
##########

folder = "../Data/HSE-2/6/"
df1 = pd.read_csv(folder+"6_1_comp.csv")
df2 = pd.read_csv(folder+"6_2_comp.csv")
df3 = pd.read_csv(folder+"6_3_direct.csv")
df4 = pd.read_csv(folder+"6_4_direct.csv")
df5 = pd.read_csv(folder+"6_5_comp.csv")

dfd = pd.read_csv(folder+"6_direct_final.csv")
dfc = pd.read_csv(folder+"6_comp_final.csv")

results_1d = []
results_1c = []

results_1d.append(getResults(df3, df4, "3", "4", "Direct"))
results_1d.append(getResults(df3, dfd, "3", "Group", "Direct"))
results_1d.append(getResults(df4, dfd, "4", "Group", "Direct"))

results_1c.append(getResults(df1, df2, "1", "2", "Comp"))
results_1c.append(getResults(df1, df5, "1", "5", "Comp"))
results_1c.append(getResults(df2, df5, "2", "5", "Comp"))
results_1c.append(getResults(df1, dfc, "1", "Group", "Comp"))
results_1c.append(getResults(df2, dfc, "2", "Group", "Comp"))
results_1c.append(getResults(df5, dfc, "5", "Group", "Comp"))

df_g_1d = pd.DataFrame(results_1d)
print("Group-6, Direct")
print(df_g_1d)
print()

df_g_1c = pd.DataFrame(results_1c)
df_g_1c = df_g_1c.dropna(axis=1,how='all')
print("Group-6, Comparative")
print(df_g_1c)
print()

results_q = []
results_q.append({"P": "1", "Type": "Comp", "Time": 148/(len(df1.index)-1)})
results_q.append({"P": "2", "Type": "Direct", "Time": 129/(len(df2.index)-2)})
results_q.append({"P": "3", "Type": "Comp", "Time": 103/(len(df3.index)-1)})
results_q.append({"P": "4", "Type": "Direct", "Time": 83/(len(df4.index)-1)})
results_q.append({"P": "5", "Type": "Comp", "Time": 144/(len(df4.index)-1)})
results_q.append({"P": "Group", "Type": "Comp", "Time": 205/(len(dfc.index)-2)})
results_q.append({"P": "Group", "Type": "Direct", "Time": 307/(len(dfc.index)-1)})

df_g_1q = pd.DataFrame(results_q)
print("Group-6, times")
print(df_g_1q)