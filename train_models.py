"""
Phishing Website Detection using Machine Learning
=================================================
Final Year Project - University of Benin
Dataset: UCI Phishing Website Dataset (11,055 instances, 30 features)

Algorithms implemented (as used across all 7 reference papers):
  1. Logistic Regression (LR)
  2. Decision Tree (DT)
  3. Random Forest (RF)        <- Best performer in most papers
  4. K-Nearest Neighbors (KNN)
  5. Support Vector Machine (SVM)
  6. AdaBoost
  7. XGBoost                   <- Tied best with RF in several papers
  8. Naive Bayes (NB)
  9. Extra Trees (ET)
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, matthews_corrcoef)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                               ExtraTreesClassifier, BaggingClassifier,
                               VotingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data',    'phishing_website_uci.csv')
MDL_DIR   = os.path.join(BASE, 'models')
OUT_DIR   = os.path.join(BASE, 'outputs')
os.makedirs(MDL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────
print("=" * 65)
print("  PHISHING WEBSITE DETECTION — ML TRAINING PIPELINE")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"\n[DATA]  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

X = df.drop('Result', axis=1)
y = df['Result']

# Map labels: -1 (phishing) → 0,  1 (legitimate) → 1  (sklearn convention)
y = y.map({-1: 0, 1: 1})
print(f"         Phishing  : {(y==0).sum():,}  ({(y==0).mean()*100:.1f}%)")
print(f"         Legitimate: {(y==1).sum():,}  ({(y==1).mean()*100:.1f}%)")

# Train / test split  (80 / 20) — same as Piserchia 2025, Latif et al. 2025
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"\n[SPLIT] Train: {len(X_train):,}   Test: {len(X_test):,}  (80/20)")

# ─────────────────────────────────────────────
# 2. DEFINE MODELS
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)":           SVC(kernel='rbf', probability=True, random_state=42),
    "Naive Bayes":         GaussianNB(),
    "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='logloss', verbosity=0),
    "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=42),
}

# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE
# ─────────────────────────────────────────────
print("\n[TRAINING]  Running 9 algorithms …\n")

results = []
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t1 = time.time()
    y_pred  = model.predict(X_test)
    test_time = time.time() - t1

    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')

    # 10-fold CV accuracy
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    cv_acc = cv_scores.mean()

    results.append({
        "Algorithm":    name,
        "Accuracy (%)": round(acc * 100, 2),
        "Precision":    round(prec, 4),
        "Recall":       round(rec, 4),
        "F1-Score":     round(f1, 4),
        "MCC":          round(mcc, 4),
        "ROC-AUC":      round(auc, 4),
        "CV-10 Acc(%)": round(cv_acc * 100, 2),
        "Train Time(s)":round(train_time, 3),
        "Test Time(s)": round(test_time, 4),
    })

    # Save model
    joblib.dump(model, os.path.join(MDL_DIR, f"{name.replace(' ', '_').replace('(','').replace(')','')}.pkl"))
    print(f"  ✓ {name:<22}  Acc={acc*100:.2f}%  F1={f1:.4f}  AUC={auc:.4f}  CV={cv_acc*100:.2f}%")

# ─────────────────────────────────────────────
# 4. RESULTS TABLE
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("Accuracy (%)", ascending=False)
print("\n" + "=" * 65)
print("  RESULTS SUMMARY (sorted by Accuracy)")
print("=" * 65)
print(results_df[["Algorithm","Accuracy (%)","Precision","Recall",
                  "F1-Score","MCC","ROC-AUC","CV-10 Acc(%)"]].to_string(index=False))

results_df.to_csv(os.path.join(OUT_DIR, 'model_results.csv'), index=False)
print(f"\n[SAVED] Results → outputs/model_results.csv")

# ─────────────────────────────────────────────
# 5. BEST MODEL — CONFUSION MATRIX
# ─────────────────────────────────────────────
best_name  = results_df.iloc[0]["Algorithm"]
best_model = joblib.load(os.path.join(MDL_DIR,
    f"{best_name.replace(' ', '_').replace('(','').replace(')','')}.pkl"))
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Phishing', 'Legitimate'],
            yticklabels=['Phishing', 'Legitimate'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix — {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix_best.png'), dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 6. ACCURACY BAR CHART
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71' if a == results_df["Accuracy (%)"].max() else '#3498db'
          for a in results_df["Accuracy (%)"]]
bars = ax.barh(results_df["Algorithm"], results_df["Accuracy (%)"], color=colors)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Phishing Detection — Algorithm Accuracy Comparison")
ax.set_xlim([85, 100])
for bar, val in zip(bars, results_df["Accuracy (%)"]):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
            f"{val:.2f}%", va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'accuracy_comparison.png'), dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE (from Random Forest)
# ─────────────────────────────────────────────
rf_model = joblib.load(os.path.join(MDL_DIR, 'Random_Forest.pkl'))
feat_imp  = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
feat_imp.head(15).plot(kind='barh', ax=ax, color='#e74c3c')
ax.invert_yaxis()
ax.set_xlabel("Feature Importance Score")
ax.set_title("Top 15 Most Important Features (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'feature_importance.png'), dpi=150)
plt.close()

# Save feature importance
feat_imp.reset_index().rename(columns={'index':'Feature', 0:'Importance'}).to_csv(
    os.path.join(OUT_DIR, 'feature_importance.csv'), index=False)

# ─────────────────────────────────────────────
# 8. MULTI-METRIC COMPARISON CHART
# ─────────────────────────────────────────────
metrics = ['Accuracy (%)', 'Precision', 'Recall', 'F1-Score']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, metric in zip(axes.flatten(), metrics):
    vals = results_df[metric] if metric != 'Accuracy (%)' else results_df[metric] / 100
    ax.barh(results_df['Algorithm'], vals, color='#9b59b6')
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} by Algorithm")
    ax.set_xlim([0.8, 1.01])
plt.suptitle("Multi-Metric Comparison — Phishing Detection Models", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'multi_metric_comparison.png'), dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 9. SAVE BEST MODEL  (for predict.py)
# ─────────────────────────────────────────────
joblib.dump(best_model, os.path.join(MDL_DIR, 'best_model.pkl'))
joblib.dump(list(X.columns), os.path.join(MDL_DIR, 'feature_names.pkl'))

print(f"\n[BEST MODEL]  → {best_name}")
print(f"[SAVED]  best_model.pkl  |  feature_names.pkl")
print("\n[CHARTS SAVED TO outputs/]")
print("  • confusion_matrix_best.png")
print("  • accuracy_comparison.png")
print("  • feature_importance.png")
print("  • multi_metric_comparison.png")
print("\n" + "=" * 65)
print("  TRAINING COMPLETE")
print("=" * 65)
