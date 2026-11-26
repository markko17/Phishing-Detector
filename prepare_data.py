"""
STEP 2: Prepare the data (now with 2 datasets)
----------------------------------------------
Inputs : data/PhiUSIIL_Phishing_URL_Dataset.csv   (label 1 = LEGITIMATE)
         data/dataset2.csv                        (Label 1 = PHISHING, newer URLs)
Output : data/clean_domains.csv  with columns:
           domain, is_phishing (1 = phishing), source, split (train/test)

Why we reduce URLs to domains:
  In PhiUSIIL, EVERY legitimate URL looks like "https://www.site.com"
  (always https, always www, never a path). If we trained on full URLs,
  the model would cheat by learning "has a path or no www = phishing".
  Using only the bare domain removes those formatting clues.

Why the train/test split is decided HERE:
  Every later script (training, evaluation, Random Forest comparison)
  reads the same "split" column, so they all test on exactly the same
  hidden 20% of domains.

Run:  python prepare_data.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from detector import SHARED_HOSTS, extract_domain

OUTPUT_PATH = "data/clean_domains.csv"
RANDOM_STATE = 42

# --------------------------------------------------- 1. Load each dataset
# Each one is converted to the SAME format: domain, is_phishing, source.
phiusiil = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv",
                       usecols=["URL", "label"])
phiusiil = pd.DataFrame({
    "url": phiusiil["URL"],
    "is_phishing": 1 - phiusiil["label"],      # flip: this dataset uses 1 = legit
    "source": "PhiUSIIL",
})

dataset2 = pd.read_csv("data/dataset2.csv")
dataset2 = pd.DataFrame({
    "url": dataset2["URL"],
    "is_phishing": dataset2["Label"],          # already 1 = phishing
    "source": "Dataset2",
})

df = pd.concat([phiusiil, dataset2], ignore_index=True)
print("Rows loaded:")
print(df["source"].value_counts().to_string(), "\n")

# --------------------------------------------------- 2. URL -> domain
df["domain"] = df["url"].apply(extract_domain)
df = df[df["domain"] != ""]

# --------------------------------------------------- 3. Remove shared platforms
# docs.google.com, bit.ly, ipfs.io ... host BOTH safe and scam pages under the
# same name, so the name alone can't tell them apart. Training on them would
# teach the AI that e.g. every Google Docs link is phishing.
before = len(df)
df = df[~df["domain"].isin(SHARED_HOSTS)]
print(f"Rows on shared platforms/shorteners removed: {before - len(df)}")

# --------------------------------------------------- 4. Remove contradictions
labels_per_domain = df.groupby("domain")["is_phishing"].nunique()
conflicting = labels_per_domain[labels_per_domain > 1].index
df = df[~df["domain"].isin(conflicting)]
print(f"Domains with conflicting labels removed: {len(conflicting)}")

# --------------------------------------------------- 5. One row per domain
# Duplicates are removed ACROSS both datasets, so the same domain can never
# be in both the training set and the test set (prevents data leakage).
# If a domain is in both datasets, the PhiUSIIL row is kept (loaded first).
before = len(df)
df = df.drop_duplicates(subset="domain")
print(f"Duplicate domains removed: {before - len(df)}")

# --------------------------------------------------- 6. Fixed train/test split
# 80% train, 20% test, keeping the same mix of source + label in both parts.
df = df[["domain", "is_phishing", "source"]].reset_index(drop=True)
strata = df["source"] + "_" + df["is_phishing"].astype(str)
train_idx, test_idx = train_test_split(df.index, test_size=0.2,
                                       stratify=strata, random_state=RANDOM_STATE)
df["split"] = "train"
df.loc[test_idx, "split"] = "test"

df.to_csv(OUTPUT_PATH, index=False)

# --------------------------------------------------- 7. Summary
print(f"\nClean domains: {len(df)}")
summary = pd.crosstab(df["source"], df["is_phishing"].map({0: "Legitimate", 1: "Phishing"}),
                      margins=True, margins_name="Total")
print(summary.to_string())
print(f"\nTrain: {(df['split'] == 'train').sum()} | Test: {(df['split'] == 'test').sum()}")
print(f"Saved to {OUTPUT_PATH}")