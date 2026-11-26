"""
STEP 3: Train and compare models
--------------------------------
Input : data/clean_domains.csv          (from Step 2)
Output: model/phishing_model.joblib      (the best model, used by the website)
        results/model_comparison.csv     (table for your paper)

How the AI "reads" a domain:
  A model can't read text directly, it needs numbers. We cut each domain
  into small overlapping pieces of 2 to 5 characters (called "character
  n-grams"). For example "paypal-login.xyz" contains pieces like
  "pa", "ay", "pal-", "-log", "login", ".xyz" ...
  TF-IDF then gives each piece a score: pieces that are common in one
  domain but rare overall get higher scores. Phishing domains tend to
  share pieces like "-login", "secure", "verify", ".web.app", random
  digit strings, etc., and the model learns which pieces signal phishing.

We compare 3 classic models on the SAME features:
  - Multinomial Naive Bayes  (simple probability model, very fast)
  - Logistic Regression      (learns a weight for every piece)
  - Linear SVM               (finds the best boundary between the 2 classes)

Run:  python train_model.py
"""

import os
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_PATH = "data/clean_domains.csv"
MODEL_PATH = "model/phishing_model.joblib"
RESULTS_PATH = "results/model_comparison.csv"
RANDOM_STATE = 42  # fixed seed so anyone re-running this gets the same results

os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Load the clean data. The 80/20 train/test split was already decided in
#    prepare_data.py (the "split" column), so every script uses the same one.
df = pd.read_csv(DATA_PATH)
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
X_train, y_train = train["domain"], train["is_phishing"]
X_test, y_test = test["domain"], test["is_phishing"]
print(f"Training set: {len(X_train)} domains | Test set: {len(X_test)} domains\n")


def make_vectorizer():
    """Turns domain text into numbers (character n-grams + TF-IDF)."""
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                           min_df=3, sublinear_tf=True)


# 3. The 3 models. Each one is a Pipeline = vectorizer + classifier,
#    so the saved model can take a raw domain string directly.
#    LinearSVC can't give probabilities on its own, so we wrap it in
#    CalibratedClassifierCV to get a confidence score (e.g. "92% phishing").
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(C=5, max_iter=2000),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(C=0.5), cv=3),
}

results = []
pipelines = {}
for name, classifier in models.items():
    pipe = Pipeline([("tfidf", make_vectorizer()), ("clf", classifier)])
    start = time.time()
    pipe.fit(X_train, y_train)
    seconds = time.time() - start
    pred = pipe.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Train time (s)": seconds,
    })
    pipelines[name] = pipe
    print(f"Trained {name} in {seconds:.1f}s")

# 4. Compare
table = pd.DataFrame(results).round(4)
print("\n" + table.to_string(index=False))
table.to_csv(RESULTS_PATH, index=False)

# 5. Pick the best model by F1 score and save it
best_name = table.loc[table["F1"].idxmax(), "Model"]
joblib.dump(pipelines[best_name], MODEL_PATH)
print(f"\nBest model (highest F1): {best_name}")
print(f"Saved to {MODEL_PATH}")

# 5b. How does the best model do on EACH dataset's test domains?
print(f"\n{best_name} - results per dataset (test set):")
best = pipelines[best_name]
for source, part in test.groupby("source"):
    pred = best.predict(part["domain"])
    print(f"  {source:<10} n={len(part):>6}  "
          f"Accuracy {accuracy_score(part['is_phishing'], pred):.4f}  "
          f"Precision {precision_score(part['is_phishing'], pred):.4f}  "
          f"Recall {recall_score(part['is_phishing'], pred):.4f}  "
          f"F1 {f1_score(part['is_phishing'], pred):.4f}")

# 6. Quick sanity test on some example domains
examples = ["bdo.com.ph", "gcash.com", "shopee.ph",
            "secure-bdo-online.weebly.com", "free-gcash-promo.firebaseapp.com",
            "paypal-verify-account.xyz"]
print("\nExample predictions (probability of phishing):")
for domain, p in zip(examples, pipelines[best_name].predict_proba(examples)[:, 1]):
    print(f"  {domain:<35} {p:6.1%}")