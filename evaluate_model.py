"""
STEP 4: Evaluate the saved model in detail
------------------------------------------
Input : data/clean_domains.csv, model/phishing_model.joblib
Output (all in results/):
  confusion_matrix.png      - what the model got right / wrong
  roc_curve.png             - how well it separates the 2 classes
  top_ngrams.png            - which character pieces the AI learned
  cross_validation.csv      - 5-fold cross-validation scores
  wrong_predictions.csv     - examples of mistakes (for your Discussion)

Run:  python evaluate_model.py
"""

import joblib
import matplotlib
matplotlib.use("Agg")  # save charts to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

DATA_PATH = "data/clean_domains.csv"
MODEL_PATH = "model/phishing_model.joblib"
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
X, y = df["domain"], df["is_phishing"]
model = joblib.load(MODEL_PATH)

# Use the same hidden 20% test set as Step 3 (the "split" column from
# prepare_data.py), so we evaluate on domains the model has never seen.
test = df[df["split"] == "test"]
X_test, y_test = test["domain"], test["is_phishing"]
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------ 1. Classification report
print("=== Classification report (test set) ===")
print(classification_report(y_test, pred,
                            target_names=["Legitimate", "Phishing"], digits=4))

# ------------------------------------------------ 2. Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["Legitimate", "Phishing"],
    cmap="Blues", values_format=",",
)
plt.title("Confusion Matrix - Linear SVM (test set)")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()

# ------------------------------------------------ 3. ROC curve
RocCurveDisplay.from_predictions(y_test, prob, name="Linear SVM")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.title("ROC Curve (test set)")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png", dpi=150)
plt.close()

# ------------------------------------------------ 4. What did the AI learn?
# The SVM gives every character piece a weight:
#   positive weight -> pushes toward PHISHING, negative -> toward LEGITIMATE.
# (The calibrated model contains 3 SVMs, so we average their weights.)
vocab = np.array(model.named_steps["tfidf"].get_feature_names_out())
weights = np.mean([c.estimator.coef_[0]
                   for c in model.named_steps["clf"].calibrated_classifiers_], axis=0)
order = np.argsort(weights)
top_phish, top_legit = order[-15:], order[:15]

fig, axes = plt.subplots(1, 2, figsize=(11, 6))
axes[0].barh([repr(v) for v in vocab[top_phish]], weights[top_phish], color="#c0392b")
axes[0].set_title("Pieces that signal PHISHING")
axes[1].barh([repr(v) for v in vocab[top_legit][::-1]], weights[top_legit][::-1], color="#2e86c1")
axes[1].set_title("Pieces that signal LEGITIMATE")
for ax in axes:
    ax.set_xlabel("Weight")
plt.tight_layout()
plt.savefig("results/top_ngrams.png", dpi=150)
plt.close()

# ------------------------------------------------ 5. 5-fold cross-validation
# Instead of trusting ONE train/test split, we split the data 5 different
# ways and train/test 5 times. If the scores are similar every time,
# the result is stable and not just luck.
print("Running 5-fold cross-validation (about 1 minute)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scores = cross_validate(clone(model), X, y, cv=cv,
                        scoring=["accuracy", "precision", "recall", "f1"], n_jobs=-1)
cv_table = pd.DataFrame({
    "Fold": [f"Fold {i}" for i in range(1, 6)],
    "Accuracy": scores["test_accuracy"],
    "Precision": scores["test_precision"],
    "Recall": scores["test_recall"],
    "F1": scores["test_f1"],
})
summary = cv_table.drop(columns="Fold").agg(["mean", "std"])
summary.insert(0, "Fold", ["Mean", "Std. dev."])
cv_table = pd.concat([cv_table, summary], ignore_index=True).round(4)
print(cv_table.to_string(index=False))
cv_table.to_csv("results/cross_validation.csv", index=False)

# ------------------------------------------------ 6. Mistakes, for discussion
wrong = pd.DataFrame({"domain": X_test, "source": test["source"], "actual": y_test,
                      "predicted": pred, "phishing_probability": prob.round(3)})
wrong = wrong[wrong["actual"] != wrong["predicted"]]
wrong["error_type"] = np.where(wrong["predicted"] == 1,
                               "False positive (safe site flagged)",
                               "False negative (phishing missed)")
wrong.to_csv("results/wrong_predictions.csv", index=False)
print(f"\nMistakes on test set: {len(wrong)} of {len(X_test)}")
print(wrong["error_type"].value_counts().to_string())

print("\nSaved charts and tables to the results/ folder.")