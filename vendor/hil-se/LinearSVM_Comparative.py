import numpy as np
import scipy
import os
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
import pandas as pd
import tensorflow as tf
from pdb import set_trace
import time

def loadData(dataName="appceleratorstudio", datatype="train"):
    path = "../Data/GPT2SP Data/Split/"
    df = pd.read_csv(path+dataName+"_"+datatype+".csv")
    return df

model = SentenceTransformer("all-MiniLM-L6-v2")

def process(dataName="appceleratorstudio", labelName="Storypoint"):
    train = loadData(dataName=dataName, datatype="train")
    valid = loadData(dataName=dataName, datatype="val")
    test = loadData(dataName=dataName, datatype="test")

    train_list_df = train.sample(frac=1)
    embeddings = model.encode(train_list_df["Issue"]) # Generate SBERT embeddings
    train_list = pd.DataFrame([{"A": embeddings[i], "Score": train_list_df[labelName][i]} for i in range(len(train))]) # Format into Embeddings - Storypoint format

    val_list_df = valid.sample(frac=1)
    embeddings = model.encode(val_list_df["Issue"])
    val_list = pd.DataFrame([{"A": embeddings[i], "Score": val_list_df[labelName][i]} for i in range(len(valid))])

    test_list_df = test.sample(frac=1)
    embeddings = model.encode(test_list_df["Issue"])
    test_list = pd.DataFrame([{"A": embeddings[i], "Score": test_list_df[labelName][i]} for i in range(len(test))])

    return train_list, val_list, test_list

def generate_comparative_judgments(train_list, N=1): # Generating pairwise data
    m = len(train_list)
    train_list.index = range(m)
    features = {"A": [], "B": [], "Label": []}
    seen = set() # Making sure there are no duplicate pairs
    for i in range(m):
        n = 0
        while n < N:
            j = np.random.randint(0, m)
            if (i,j) in seen or (j,i) in seen:
                continue
            if train_list["Score"][i] > train_list["Score"][j]:
                features["A"].append(train_list["A"][i])
                features["B"].append(train_list["A"][j])
                features["Label"].append(1.0)
                n += 1
            elif train_list["Score"][i] < train_list["Score"][j]:
                features["A"].append(train_list["A"][i])
                features["B"].append(train_list["A"][j])
                features["Label"].append(-1.0)
                n += 1
            seen.add((i, j))
    features = {key: np.array(features[key]) for key in features}
    return features

def train_and_test_iterative(dataname, N=1, itr=20):
    train_list, val_list, test_list = process(dataname, "Storypoint")

    train_list_og = pd.concat([train_list, val_list], axis=0) # Combine training and validation sets

    test_x = np.array(test_list["A"].tolist())
    test_y = test_list["Score"].tolist()

    results = []

    for i in range(itr):
        print()
        print(dataname, N, i+1)
        print()
        train_list = train_list_og.sample(frac=1.0) # Shuffle original data
        train_list.index = range(len(train_list))
        features = generate_comparative_judgments(train_list, N=N) # Format data into pairwise format, essentially generating new pairs every iteration

        train_x = np.array(train_list["A"].tolist())
        train_y = np.array(train_list["Score"].tolist())
        train_feature = features["A"]-features["B"]

        model = LinearSVC(loss="hinge", fit_intercept = False) # Declare new SVM model
        model.fit(train_feature, features["Label"]) # Train new SVM model
        preds_test = model.decision_function(test_x).flatten() # Get predictions
        preds_train = model.decision_function(train_x).flatten()

        pearsons_train = scipy.stats.pearsonr(preds_train, train_y)[0]
        spearmans_train = scipy.stats.spearmanr(preds_train, train_y).statistic
        pearsons_test = scipy.stats.pearsonr(preds_test, test_y)[0]
        spearmans_test = scipy.stats.spearmanr(preds_test, test_y).statistic
        results.append((pearsons_train.item(), spearmans_train.item(), pearsons_test.item(), spearmans_test.item()))

    return results


# datas = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement", "duracloud", "jirasoftware",
#          "mesos", "moodle", "mule", "mulestudio", "springxd", "talenddataquality", "talendesb", "titanium", "usergrid"]
datas = ["clover"]

results = []
times = []
itrs = 20
for d in datas:
    for n in [1, 2, 3, 4, 5, 10]:
    # for n in [1]:
        start = time.time()
        temp_r = train_and_test_iterative(d, n, itrs)
        for r in temp_r:
            r_train, rs_train, r_test, rs_test = r
            print(d, r_train, rs_train, r_test, rs_test)
            results.append({"Data": d, "N": n, "Pearson Train": r_train, "Spearman Train": rs_train, "Pearson Test": r_test, "Spearman Test": rs_test})
        end = time.time()
        elapsed_time = end-start
        times.append(elapsed_time)
results = pd.DataFrame(results)
print(results)
results.to_csv("../Results/LinearSVM-Comparative.csv", index=False)

for t in times:
    print(t)

print(sum(times))
