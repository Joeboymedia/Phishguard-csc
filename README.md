# Phishing Website Detection using Machine Learning
### Final Year Project — University of Benin
**Matric No:** PSC2207888

---

## Project Overview

This system detects phishing websites using **9 machine learning algorithms** trained on the
UCI Phishing Website Dataset (11,055 samples, 30 features). It is backed by 7 peer-reviewed
papers from 2020–2025.

**Algorithms implemented:**
| # | Algorithm | Source Papers |
|---|-----------|--------------|
| 1 | Logistic Regression | Latif et al. 2025; Broștic et al. 2025 |
| 2 | Decision Tree | Jayasurya et al. 2025; Piserchia 2025 |
| 3 | Random Forest ⭐ | All 7 papers — consistently best |
| 4 | K-Nearest Neighbors | Latif et al. 2025; Jayasurya et al. 2025 |
| 5 | Support Vector Machine | Shanmugapriya et al. 2025 |
| 6 | Naive Bayes | Latif et al. 2025 |
| 7 | AdaBoost | Jayasurya et al. 2025; Alsariera et al. 2020 |
| 8 | XGBoost ⭐ | Latif et al. 2025; KJMR 2025 |
| 9 | Extra Trees | Alsariera et al. 2020 (IEEE Access) |

---

## Project Structure

```
phishing_detection/
│
├── data/
│   └── phishing_website_uci.csv       ← UCI dataset (11,055 × 31)
│
├── models/
│   ├── best_model.pkl                 ← Best trained model
│   ├── feature_names.pkl              ← Feature list
│   ├── Random_Forest.pkl
│   ├── XGBoost.pkl
│   └── ...                            ← All 9 models
│
├── outputs/
│   ├── model_results.csv              ← Full results table
│   ├── accuracy_comparison.png        ← Bar chart
│   ├── confusion_matrix_best.png      ← Confusion matrix
│   ├── feature_importance.png         ← Top 15 features
│   └── multi_metric_comparison.png    ← 4-metric chart
│
├── references/
│   ├── 01_Ensemble_ML_IJIRSET_2025.pdf
│   ├── 02_Phishing_Detection_KJMR_2025.pdf
│   ├── 03_Phishing_ML_Techniques_IJSAT_2025.pdf
│   ├── 04_RuleBasedSystem_ICRDICCT_2025.pdf
│   ├── 05_Comparative_Analysis_IRASS_2025.pdf
│   ├── 06_MetaLearners_ExtraTrees_IEEE_2020.pdf
│   └── 07_SVM_LR_RF_Romanian_CyberSecurity_2025.pdf
│
├── train_models.py                    ← Main training script
├── predict.py                         ← Prediction script
├── requirements.txt
└── README.md
```

---

## STEP-BY-STEP SETUP GUIDE

### Step 1 — Install Python
Download Python 3.10 or later from https://python.org.
During installation on Windows, check ✅ **"Add Python to PATH"**.

Verify:
```bash
python --version
```

---

### Step 2 — Set Up Project Folder
Create a folder called `phishing_detection` anywhere on your computer (e.g., Desktop).
Copy all project files into it, keeping the folder structure shown above.

---

### Step 3 — Open Terminal / Command Prompt
- **Windows:** Press `Win + R`, type `cmd`, press Enter
- **Mac/Linux:** Open Terminal

Navigate to the project folder:
```bash
cd Desktop/phishing_detection
```

---

### Step 4 — Install Dependencies
```bash
pip install -r requirements.txt
```

This installs: scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn, joblib, imbalanced-learn.

---

### Step 5 — Run the Training Script
```bash
python train_models.py
```

**What it does:**
1. Loads the dataset (11,055 website samples)
2. Trains all 9 ML algorithms
3. Evaluates using Accuracy, Precision, Recall, F1-Score, MCC, ROC-AUC, and 10-fold CV
4. Saves all models to `models/`
5. Saves results table to `outputs/model_results.csv`
6. Generates 4 charts in `outputs/`

**Expected output (example):**
```
Algorithm               Acc(%)   F1     AUC    CV(%)
XGBoost                 97.84   0.978  0.996   97.27
Random Forest           97.63   0.976  0.996   97.20
Extra Trees             97.41   0.974  0.995   97.10
...
```

**Expected run time:** 3–8 minutes depending on your machine.

---

### Step 6 — View Results
Open the `outputs/` folder. You will find:
- `model_results.csv` — Full comparison table
- `accuracy_comparison.png` — Algorithm accuracy bar chart
- `confusion_matrix_best.png` — Confusion matrix of best model
- `feature_importance.png` — Top 15 URL features ranked
- `multi_metric_comparison.png` — Precision/Recall/F1 charts

---

### Step 7 — Run Predictions on New Data
Run demo predictions:
```bash
python predict.py --demo
```

Run on your own CSV file (must have the 30 feature columns):
```bash
python predict.py --input my_urls.csv
```

Output will show: **`PHISHING ⚠️`** or **`Legitimate ✅`** with confidence %.

---

## Dataset — 30 Features Explained

| Category | Features |
|----------|----------|
| Address Bar (12) | IP Address, URL Length, Shortening Service, @ Symbol, Double Slash, Prefix-Suffix, Sub Domain, SSL State, Domain Registration Length, Favicon, Port, HTTPS Token |
| Abnormal (6) | Request URL, URL of Anchor, Links in Tags, SFH, Submit to Email, Abnormal URL |
| HTML/JS (5) | Redirect, onMouseover, RightClick, PopupWindow, IFrame |
| Domain (7) | Age of Domain, DNS Record, Web Traffic, Page Rank, Google Index, Links Pointing to Page, Statistical Report |

**Feature values:** `-1` = suspicious/phishing indicator, `0` = neutral, `1` = legitimate indicator
**Target (Result):** `-1` = Phishing, `1` = Legitimate

---

## Expected Results (from Literature)

| Algorithm | Expected Accuracy | Source |
|-----------|-----------------|--------|
| Random Forest | 97.2–99.0% | 6 of 7 papers |
| XGBoost | 97.8–99.0% | Latif et al., KJMR 2025 |
| Extra Trees | 97.4% | Alsariera et al., IEEE 2020 |
| Decision Tree | 95.9–97.8% | Multiple papers |
| Logistic Regression | 92.7–98.5% | Multiple papers |
| KNN | 91.5–95.3% | Multiple papers |
| Naive Bayes | 92.9–93.5% | Multiple papers |

---

## References (7 Papers)

1. Aarthi et al. (2025). *Phishing Website Detection using Ensemble Machine Learning Approach*. IJIRSET, Vol.14(4).
2. Latif & Pervaiz (2025). *Detecting Phishing Attacks in Cybersecurity using ML with Data Preprocessing and Feature Engineering*. KJMR, Vol.2(3).
3. Jayasurya et al. (2025). *Phishing Detection Using Machine Learning Techniques*. IJSAT, Vol.16(2).
4. Shanmugapriya et al. (2025). *Automated Phishing Website Detection and Analysis Using Advanced Machine Learning Techniques*. ICRDICCT'25.
5. Piserchia, O. (2025). *Phishing Website Detection through Machine Learning Algorithms: A Comparative Analysis*. IRASS Journal, Vol.2(12).
6. Alsariera et al. (2020). *AI Meta-Learners and Extra-Trees Algorithm for the Detection of Phishing Websites*. IEEE Access, Vol.8.
7. Broștic et al. (2025). *A Study on Detecting and Preventing Phishing Attacks Using Machine Learning Techniques*. Romanian Cyber Security Journal, Vol.7(2).
