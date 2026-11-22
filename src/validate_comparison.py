import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    root_mean_squared_error,
    mean_poisson_deviance
)
from src.train import EXCLUDED_COLS, assign_risk_category, RISK_CATEGORIES

# Load data
df = pd.read_csv("data/processed/model_dataset.csv")
feat_cols = [c for c in df.columns if c not in EXCLUDED_COLS]

val_df = df[df["split"] == "val"].copy()
X_val = val_df[feat_cols]

# Load Option A (production baseline models)
m2_a = joblib.load("models/model_2w.joblib")
m4_a = joblib.load("models/model_4w.joblib")

# Load Option B (experimental weighted models)
m2_b = joblib.load("models/experiments/model_2w_weighted.joblib")
m4_b = joblib.load("models/experiments/model_4w_weighted.joblib")

# Actual ground truth categories
y_v2 = val_df["target_2w"].values
y_v4 = val_df["target_4w"].values
act_v2 = assign_risk_category(y_v2)
act_v4 = assign_risk_category(y_v4)

# Continuous predictions
p2_a = np.clip(m2_a.predict(X_val), 0, None)
p4_a = np.clip(m4_a.predict(X_val), 0, None)
p2_b = np.clip(m2_b.predict(X_val), 0, None)
p4_b = np.clip(m4_b.predict(X_val), 0, None)

def apply_thresholds(preds, thresholds):
    tl, tm, tc = thresholds
    cats = np.empty(len(preds), dtype=object)
    cats[preds < tl] = "Low"
    cats[(preds >= tl) & (preds < tm)] = "Medium"
    cats[(preds >= tm) & (preds < tc)] = "High"
    cats[preds >= tc] = "Critical"
    return cats

# Option A calibrated thresholds:
# 2-week: Low < 0.8, Medium: 0.8 to < 4.0, High: 4.0 to < 8.0, Critical: >= 8.0
# 4-week: Low < 0.8, Medium: 0.8 to < 4.0, High: 4.0 to < 8.5, Critical: >= 8.5
thresh_2w_a = (0.8, 4.0, 8.0)
thresh_4w_a = (0.8, 4.0, 8.5)

# Option B thresholds:
thresh_2w_b = (0.8, 3.5, 8.5)
thresh_4w_b = (0.8, 4.0, 8.5)

pred_v2_a = apply_thresholds(p2_a, thresh_2w_a)
pred_v4_a = apply_thresholds(p4_a, thresh_4w_a)
pred_v2_b = apply_thresholds(p2_b, thresh_2w_b)
pred_v4_b = apply_thresholds(p4_b, thresh_4w_b)

def print_comparison(name, y_true, act_cats, preds_a, preds_b, y_pred_cont_a, y_pred_cont_b, thresh_a):
    print("=" * 72)
    print(f"  VALIDATION EVALUATION: {name} (N = {len(y_true):,})")
    print(f"  Option A Thresholds: Low < {thresh_a[0]}, Med {thresh_a[0]}-{thresh_a[1]}, High {thresh_a[1]}-{thresh_a[2]}, Crit >= {thresh_a[2]}")
    print("=" * 72)

    mae_a, rmse_a = mean_absolute_error(y_true, y_pred_cont_a), root_mean_squared_error(y_true, y_pred_cont_a)
    mae_b, rmse_b = mean_absolute_error(y_true, y_pred_cont_b), root_mean_squared_error(y_true, y_pred_cont_b)

    f1_a = f1_score(act_cats, preds_a, labels=RISK_CATEGORIES, average="macro")
    f1_b = f1_score(act_cats, preds_b, labels=RISK_CATEGORIES, average="macro")

    cr_a = classification_report(act_cats, preds_a, labels=RISK_CATEGORIES, output_dict=True, zero_division=0)
    cr_b = classification_report(act_cats, preds_b, labels=RISK_CATEGORIES, output_dict=True, zero_division=0)

    cm_a = confusion_matrix(act_cats, preds_a, labels=RISK_CATEGORIES)
    cm_b = confusion_matrix(act_cats, preds_b, labels=RISK_CATEGORIES)

    print("Continuous Accuracy:")
    print(f"  Option A: MAE = {mae_a:.4f}, RMSE = {rmse_a:.4f}")
    print(f"  Option B: MAE = {mae_b:.4f}, RMSE = {rmse_b:.4f}")
    print(f"\nMacro F1-Score: Option A = {f1_a:.4f}  vs  Option B = {f1_b:.4f}")

    # Class metrics table
    metrics_summary = []
    for cat in RISK_CATEGORIES:
        metrics_summary.append({
            "Risk Tier": cat,
            "Opt A Recall": f"{cr_a[cat]['recall']*100:.1f}%",
            "Opt A Precision": f"{cr_a[cat]['precision']*100:.1f}%",
            "Opt A F1": f"{cr_a[cat]['f1-score']:.4f}",
            "Opt B Recall": f"{cr_b[cat]['recall']*100:.1f}%",
            "Opt B Precision": f"{cr_b[cat]['precision']*100:.1f}%",
            "Opt B F1": f"{cr_b[cat]['f1-score']:.4f}",
        })
    print("\nPer-Class Breakdown:")
    print(pd.DataFrame(metrics_summary).to_string(index=False))

    # False Critical Alarms Analysis
    crit_pred_a = (preds_a == "Critical")
    crit_pred_b = (preds_b == "Critical")
    crit_true = (act_cats == "Critical")

    tp_a = (crit_pred_a & crit_true).sum()
    fp_a = (crit_pred_a & ~crit_true).sum()
    fp_a_from_high = (crit_pred_a & (act_cats == "High")).sum()
    fp_a_from_med = (crit_pred_a & (act_cats == "Medium")).sum()
    fp_a_from_low = (crit_pred_a & (act_cats == "Low")).sum()

    tp_b = (crit_pred_b & crit_true).sum()
    fp_b = (crit_pred_b & ~crit_true).sum()
    fp_b_from_high = (crit_pred_b & (act_cats == "High")).sum()
    fp_b_from_med = (crit_pred_b & (act_cats == "Medium")).sum()
    fp_b_from_low = (crit_pred_b & (act_cats == "Low")).sum()

    print("\nCritical Alarm Analysis:")
    print(f"  Total Actual Criticals in validation: {crit_true.sum():,}")
    print("  --- Option A ---")
    print(f"    Total Predicted Critical:     {crit_pred_a.sum():,}")
    print(f"    True Critical Catches (TP):   {tp_a:,}  ({tp_a/crit_true.sum()*100:.1f}% Recall)")
    print(f"    False Critical Alarms (FP):   {fp_a:,}  ({tp_a/(tp_a+fp_a)*100:.1f}% Precision)")
    print(f"      * False Alarms from High (actual 4-9.9 cases):   {fp_a_from_high:,} ({fp_a_from_high/fp_a*100 if fp_a>0 else 0:.1f}% of all FPs)")
    print(f"      * False Alarms from Medium (actual 1-3.9 cases): {fp_a_from_med:,}")
    print(f"      * False Alarms from Low (actual 0 cases):        {fp_a_from_low:,}")
    print("  --- Option B ---")
    print(f"    Total Predicted Critical:     {crit_pred_b.sum():,}")
    print(f"    True Critical Catches (TP):   {tp_b:,}  ({tp_b/crit_true.sum()*100:.1f}% Recall)")
    print(f"    False Critical Alarms (FP):   {fp_b:,}  ({tp_b/(tp_b+fp_b)*100:.1f}% Precision)")
    print(f"      * False Alarms from High (actual 4-9.9 cases):   {fp_b_from_high:,}")
    print(f"      * False Alarms from Medium (actual 1-3.9 cases): {fp_b_from_med:,}")
    print(f"      * False Alarms from Low (actual 0 cases):        {fp_b_from_low:,}")

    print("\nConfusion Matrix - Option A (Validation):")
    cm_df_a = pd.DataFrame(cm_a, index=[f"Act_{c}" for c in RISK_CATEGORIES], columns=[f"Pred_{c}" for c in RISK_CATEGORIES])
    print(cm_df_a)
    print("\nConfusion Matrix - Option B (Validation):")
    cm_df_b = pd.DataFrame(cm_b, index=[f"Act_{c}" for c in RISK_CATEGORIES], columns=[f"Pred_{c}" for c in RISK_CATEGORIES])
    print(cm_df_b)
    print("\n")

print_comparison("2-WEEK HORIZON", y_v2, act_v2, pred_v2_a, pred_v2_b, p2_a, p2_b, thresh_2w_a)
print_comparison("4-WEEK HORIZON", y_v4, act_v4, pred_v4_a, pred_v4_b, p4_a, p4_b, thresh_4w_a)
