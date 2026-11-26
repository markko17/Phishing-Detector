"""
EXTRA: Why Linear SVM instead of Random Forest?
-----------------------------------------------
Trains Random Forest on the SAME data split as Step 3 and compares it with
the saved Linear SVM. Takes several minutes (Random Forest is slow on text).

Run:  python compare_random_forest.py
Output: results/random_forest_comparison.csv
"""

import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
df = pd.read_csv("data/clean_domains.csv")
train = df[df["split"] == "train"]   # same split as every other script
test = df[df["split"] == "test"]
X_train, y_train = train["domain"], train["is_phishing"]
X_test, y_test = test["domain"], test["is_phishing"]


def scores(name, model, train_time):
    pred = model.predict(X_test)
    joblib.dump(model, "model/_tmp_size_check.joblib")
    size_mb = os.path.getsize("model/_tmp_size_check.joblib") / 1e6
    os.remove("model/_tmp_size_check.joblib")
    return {"Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "Train time (s)": train_time,
            "Model size (MB)": size_mb}


rows = []

# Linear SVM (already trained in Step 3). Retrain once just to time it fairly.
svm = joblib.load("model/phishing_model.joblib")
start = time.time()
svm_fresh = Pipeline(svm.steps).fit(X_train, y_train)
rows.append(scores("Linear SVM (all n-grams)", svm_fresh, time.time() - start))

# Random Forest. With all ~268,000 n-gram features it runs out of memory on
# a normal laptop, so we limit it to the 20,000 most common n-grams.
print("Training Random Forest (this takes several minutes)...")
rf = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3,
                              sublinear_tf=True, max_features=20000)),
    ("clf", RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                   random_state=RANDOM_STATE)),
])
start = time.time()
rf.fit(X_train, y_train)
rows.append(scores("Random Forest (20k n-grams, 100 trees)", rf, time.time() - start))

table = pd.DataFrame(rows).round(4)
print(table.to_string(index=False))
table.to_csv("results/random_forest_comparison.csv", index=False)
print("\nSaved to results/random_forest_comparison.csv")