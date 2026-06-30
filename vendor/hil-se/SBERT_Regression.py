import numpy as np
import scipy
import os
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVR
import sklearn
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
    embeddings = model.encode(train_list_df["Issue"])
    train_list = pd.DataFrame([{"A": embeddings[i], "Score": train_list_df[labelName][i]} for i in range(len(train))])

    val_list_df = valid.sample(frac=1)
    embeddings = model.encode(val_list_df["Issue"])
    val_list = pd.DataFrame([{"A": embeddings[i], "Score": val_list_df[labelName][i]} for i in range(len(valid))])

    test_list_df = test.sample(frac=1)
    embeddings = model.encode(test_list_df["Issue"])
    test_list = pd.DataFrame([{"A": embeddings[i], "Score": test_list_df[labelName][i]} for i in range(len(test))])

    return train_list, val_list, test_list


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(1, activation="linear")
    ])

    model.compile(
        optimizer='adam',
        loss="mae",
        metrics=['mae']
    )

    return model

def train_and_test_iterative(dataname, itr):
    print(dataname)
    train_list_og, val_list_og, test_list = process(dataname, "Storypoint")
    train_list_og = pd.concat([train_list_og, val_list_og], axis=0)

    test_x = np.array(test_list["A"].tolist())
    test_y = test_list["Score"].tolist()

    results = []
    for i in range(itr):
        print()
        print(dataname, i+1)
        print()

        train_list = train_list_og.sample(frac=1.0)
        train_x = np.array(train_list["A"].tolist())
        train_y = np.array(train_list["Score"].tolist())


        model = build_model((train_x.shape[1]))

        checkpoint_path = "checkpoint/regression.keras"
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                         patience=10,
                                                         factor=0.3,
                                                         min_lr=1e-6,
                                                         verbose=1)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        monitor="loss",
                                                        save_best_only=True,
                                                        save_weights_only=True,
                                                        verbose=1)

        history = model.fit(train_x, train_y,
                            validation_data=None,
                            batch_size=32,
                            epochs=1000,
                            callbacks=[reduce_lr, checkpoint],
                            verbose=1)

        print("\nLoading best checkpoint model...")
        model.load_weights(checkpoint_path)

        preds_test = model.predict(test_x).flatten()
        preds_train = model.predict(train_x).flatten()

        pearsons_train = scipy.stats.pearsonr(preds_train, train_y)[0]
        spearmans_train = scipy.stats.spearmanr(preds_train, train_y).statistic
        pearsons_test = scipy.stats.pearsonr(preds_test, test_y)[0]
        spearmans_test = scipy.stats.spearmanr(preds_test, test_y).statistic

        mae_test = sklearn.metrics.mean_absolute_error(preds_test, test_y)
        mae_train = sklearn.metrics.mean_absolute_error(preds_train, train_y)

        results.append((pearsons_train.item(), spearmans_train.item(), mae_train, pearsons_test.item(), spearmans_test.item(), mae_test))

    return results



datas = ["appceleratorstudio", "aptanastudio", "bamboo", "clover", "datamanagement", "duracloud", "jirasoftware",
         "mesos", "moodle", "mule", "mulestudio", "springxd", "talenddataquality", "talendesb", "titanium", "usergrid"]
# datas = ["clover"]

results = []
times = []
itrs = 20
for d in datas:
    start = time.time()
    temp_r = train_and_test_iterative(d, itrs)
    end = time.time()
    elapsed_time = (end-start)/itrs
    times.append(elapsed_time)
    for r in temp_r:
        r_train, rs_train, mae_train, r_test, rs_test, mae_test = r
        print(d, r_train, rs_train, mae_train, r_test, rs_test, mae_test)
        results.append({"Data": d, "Pearson Train": r_train, "Spearman Train": rs_train, "MAE Train": mae_train, "Pearson Test": r_test, "Spearman Test": rs_test, "MAE Test": mae_test})
results = pd.DataFrame(results)
print(results)
results.to_csv("../Results/SBERT-Regression.csv", index=False)

print("\n\nElapsed times:")
for t in times:
    print(t)
print("\n\nTotal:", sum(times))