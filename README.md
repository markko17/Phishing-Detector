# AI-Based Phishing Website Detection System

A web application that checks whether a website link is a phishing site. Paste a link, and the system shows a verdict, a phishing probability and plain-language warning signs explaining the result.

The AI is a **Linear Support Vector Machine (SVM)** trained on 251,941 website domains. It reaches **88.4% accuracy**, **93.3% precision**, **79.9% recall** and an **F1 score of 0.861** on 50,389 test domains it had never seen.

> **Note:** This is a student prototype for CpE Elective 3. It can make mistakes and should not replace caution or official security tools.

---

## Features

- **AI phishing detection** gives a phishing probability for any website domain.
- **Three risk levels:** Phishing (70% or higher), Suspicious (50–70%) and Likely legitimate (below 50%).
- **Explainable results** list warning signs such as fake brand names, risky domain endings and free website builders, and show the parts of the domain that influenced the AI.
- **Trusted domain list** prevents false alarms on well-known official sites such as BDO, GCash and Shopee.
- **Shared platform check** marks links on Google Docs, IPFS and link shorteners (bit.ly, TinyURL) as "Check carefully".
- **Adult content filter** blocks known adult websites with a separate result.
- **Recent checks history** lists every link checked in the current session.
- **Works offline:** after setup, no internet connection is needed.

---

## How it works

```
User pastes a link
      ↓
Reduce the link to its domain   (https://www.site.com/login → site.com)
      ↓
Known adult site?               → yes → Adult content: blocked
      ↓ no
Shared platform or shortener?   → yes → Check carefully
      ↓ no
On the trusted domain list?     → yes → Legitimate
      ↓ no
AI model (Linear SVM)           → Phishing / Suspicious / Likely legitimate
```

The AI reads a domain as small overlapping pieces of 2 to 5 characters (character n-grams), weighted with TF-IDF. It learned which pieces usually appear in phishing domains, such as `-login`, `verify` and `.xyz`, and which appear in legitimate ones, such as `.gov` and `.edu`.

---

## Requirements

- Windows, macOS or Linux
- Python 3.10 or newer
- About 1 GB of free disk space for the datasets

---

## Installation

**1. Open a terminal in the project folder.** On Windows, open the `phishing-detector` folder in File Explorer, click the address bar, type `powershell` and press Enter.

**2. Create and turn on a virtual environment.**

```powershell
python -m venv venv
venv\Scripts\activate
```

On macOS or Linux, use `source venv/bin/activate` instead. You should see `(venv)` at the start of the line.

> If PowerShell says "running scripts is disabled," run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

**3. Install the required libraries.**

```powershell
pip install -r requirements.txt
```

**4. Add the datasets** to the `data/` folder with these exact names:

- `data/PhiUSIIL_Phishing_URL_Dataset.csv`
- `data/dataset2.csv`

---

## Running the system

### First time only: prepare the data and train the model

```powershell
python prepare_data.py
python train_model.py
python evaluate_model.py
```

| Script | What it does | Time |
|---|---|---|
| `prepare_data.py` | Cleans both datasets into `data/clean_domains.csv` | about 30 seconds |
| `train_model.py` | Trains 3 models, compares them and saves the best one to `model/phishing_model.joblib` | about 1 minute |
| `evaluate_model.py` | Creates charts and tables in `results/` | about 1–2 minutes |

You only need to run these again if you change the data or the scripts.

### Start the website

```powershell
streamlit run app.py
```

Your browser opens at **http://localhost:8501**. To stop the website, press `Ctrl + C` in the terminal.

Every time you open a new terminal, go back to the project folder and run `venv\Scripts\activate` before starting the website.

### Try these links

| Link | Expected result |
|---|---|
| `shopee.ph` | Legitimate (trusted list) |
| `secure-bdo-online.weebly.com` | Phishing |
| `paypal-verify-account.xyz` | Phishing |
| `bit.ly/3xYz` | Check carefully |
| `uni-mainz.de` | Likely legitimate |

---

## Project structure

```
phishing-detector/
├── data/
│   ├── PhiUSIIL_Phishing_URL_Dataset.csv   original dataset 1
│   ├── dataset2.csv                        original dataset 2
│   └── clean_domains.csv                   cleaned data (created by prepare_data.py)
├── model/
│   └── phishing_model.joblib               trained AI model (created by train_model.py)
├── results/                                charts and tables (created by evaluate_model.py)
├── prepare_data.py                         Step 2: data cleaning
├── train_model.py                          Step 3: model training and comparison
├── evaluate_model.py                       Step 4: detailed evaluation
├── compare_random_forest.py                extra: Linear SVM vs Random Forest
├── detector.py                             detection logic used by the website
├── app.py                                  the Streamlit website
├── requirements.txt                        library versions
└── README.md                               this file
```

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Linear SVM (selected)** | **88.37%** | **93.30%** | 79.85% | **0.8605** |
| Logistic Regression | 88.07% | 91.86% | 80.57% | 0.8585 |
| Naive Bayes | 85.56% | 93.90% | 72.57% | 0.8187 |
| Random Forest | 87.58% | 95.30% | 76.10% | 0.8462 |

5-fold cross-validation of the Linear SVM gave an F1 of 0.860 ± 0.001, so the result is stable.

---

## Customizing the lists

Open `detector.py` and edit these lists near the top:

| List | Purpose |
|---|---|
| `TRUSTED_DOMAINS` | Official sites that are always marked legitimate. Add your school's website here. |
| `SHARED_HOSTS` | Shared platforms and link shorteners that get "Check carefully" |
| `ADULT_DOMAINS` | Known adult websites to block |
| `ADULT_EXCEPTIONS` | Normal sites that the adult filter blocks by mistake |

Restart the website after editing.

---

## Limitations

- The AI judges the **domain name only**. It does not open the page or check its content.
- Phishing pages on **hacked real websites** with normal-looking names are often missed. About 1 in 5 phishing sites can get through.
- Links on shared platforms and shorteners can't be judged by name, so they are only flagged for caution.
- Phishing tactics change, so the model should be retrained with newer data over time.

---

## Datasets

- **PhiUSIIL Phishing URL Dataset:** Prasad, A., & Chandra, S. (2024). UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
- **Dataset 2:** *[add the author, year, title and link]*

---

## Built with

Python · pandas · scikit-learn · joblib · Streamlit · matplotlib

---

## Authors

*[Lagdaan, Mark B. - Estallo, Ferdie V.]*
CpE Elective 3, 4th Year, 1st Semester
