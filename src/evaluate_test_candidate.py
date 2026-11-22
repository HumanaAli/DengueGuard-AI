import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
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

# 1. Save the calibrated thresholds configuration separately
config = {
    "strategy": "Option A: Post-hoc Threshold Calibration on Baseline HistGBM-Poisson",
    "status": "Candidate (Pending user approval)",
    "models_used": {
        "2w": "models/model_2w.joblib",
        "4w": "models/model_4w.joblib"
    },
    "thresholds": {
        "2w": {"Low_max": 0.8, "Medium_max": 4.0, "High_max": 8.0, "Critical_min": 8.0},
        "4w": {"Low_max": 0.8, "Medium_max": 4.0, "High_max": 8.5, "Critical_min": 8.5}
    },
    "actual_risk_definitions": {
        "Low": "cases < 1.0",
        "Medium": "1.0 <= cases < 4.0",
        "High": "4.0 <= cases < 10.0",
        "Critical": "cases >= 10.0"
    }
}
os.makedirs("models", exist_ok=True)
with open("models/calibrated_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("Saved models/calibrated_config.json successfully.")

# 2. Perform ONE-TIME final evaluation on untouched 2025 Test Set
df = pd.read_csv("data/processed/model_dataset.csv")
feat_cols = [c for c in df.columns if c not in EXCLUDED_COLS]
test_df = df[df["split"] == "test"].copy()
X_test = test_df[feat_cols]

m2 = joblib.load("models/model_2w.joblib")
m4 = joblib.load("models/model_4w.joblib")

y_t2 = test_df["target_2w"].values
y_t4 = test_df["target_4w"].values
act_t2 = assign_risk_category(y_t2)
act_t4 = assign_risk_category(y_t4)

p2_test = np.clip(m2.predict(X_test), 0, None)
p4_test = np.clip(m4.predict(X_test), 0, None)

def apply_thresholds(preds, thresholds):
    tl, tm, tc = thresholds
    cats = np.empty(len(preds), dtype=object)
    cats[preds < tl] = "Low"
    cats[(preds >= tl) & (preds < tm)] = "Medium"
    cats[(preds >= tm) & (preds < tc)] = "High"
    cats[preds >= tc] = "Critical"
    return cats

pred_t2 = apply_thresholds(p2_test, (0.8, 4.0, 8.0))
pred_t4 = apply_thresholds(p4_test, (0.8, 4.0, 8.5))

for name, y_true, act_cats, pred_cats, p_cont, thresh in [
    ("2-WEEK HORIZON", y_t2, act_t2, pred_t2, p2_test, (0.8, 4.0, 8.0)),
    ("4-WEEK HORIZON", y_t4, act_t4, pred_t4, p4_test, (0.8, 4.0, 8.5))
]:
    print("=" * 72)
    print(f"  FINAL UNTOUCHED TEST SET EVALUATION (2025 DATA, N = {len(y_true):,})")
    print(f"  Horizon: {name}")
    print(f"  Calibrated Thresholds: Low<{thresh[0]}, Med {thresh[0]}-{thresh[1]}, High {thresh[1]}-{thresh[2]}, Crit>={thresh[2]}")
    print("=" * 72)

    mae = mean_absolute_error(y_true, p_cont)
    rmse = root_mean_squared_error(y_true, p_cont)
    pdev = mean_poisson_deviance(y_true, np.clip(p_cont, 1e-6, None))
    macro_f1 = f1_score(act_cats, pred_cats, labels=RISK_CATEGORIES, average="macro")
    macro_prec = precision_score(act_cats, pred_cats, labels=RISK_CATEGORIES, average="macro")
    macro_rec = recall_score(act_cats, pred_cats, labels=RISK_CATEGORIES, average="macro")

    cr = classification_report(act_cats, pred_cats, labels=RISK_CATEGORIES, output_dict=True, zero_division=0)
    cm = confusion_matrix(act_cats, pred_cats, labels=RISK_CATEGORIES)

    print("Continuous Regression Metrics:")
    print(f"  MAE:              {mae:.4f} cases")
    print(f"  RMSE:             {rmse:.4f} cases")
    print(f"  Poisson Deviance: {pdev:.4f}")
    print("\nOrdinal Classification Metrics:")
    print(f"  Macro F1:         {macro_f1:.4f}")
    print(f"  Macro Precision:  {macro_prec:.4f}")
    print(f"  Macro Recall:     {macro_rec:.4f}")
    print(f"  Critical Recall:   {cr['Critical']['recall']*100:.1f}%")
    print(f"  Critical Precision:{cr['Critical']['precision']*100:.1f}%")
    print(f"  Critical F1:       {cr['Critical']['f1-score']:.4f}")
    print(f"  Medium Recall:     {cr['Medium']['recall']*100:.1f}%")
    print(f"  Medium Precision:  {cr['Medium']['precision']*100:.1f}%")
    print(f"  Medium F1:         {cr['Medium']['f1-score']:.4f}")
    print(f"  High Recall:       {cr['High']['recall']*100:.1f}%")
    print(f"  Low Recall:        {cr['Low']['recall']*100:.1f}%")

    crit_pred = (pred_cats == "Critical")
    crit_true = (act_cats == "Critical")
    tp = (crit_pred & crit_true).sum()
    fp = (crit_pred & ~crit_true).sum()
    fp_high = (crit_pred & (act_cats == "High")).sum()
    fp_med = (crit_pred & (act_cats == "Medium")).sum()
    fp_low = (crit_pred & (act_cats == "Low")).sum()

    print(f"\nCritical Alarm Breakdown on Test Set:")
    print(f"  Total True Critical Events: {crit_true.sum():,}")
    print(f"  Total Predicted Critical:   {crit_pred.sum():,}")
    print(f"  True Positives (Catches):   {tp:,} ({tp/crit_true.sum()*100:.1f}% Recall)")
    print(f"  False Alarms:               {fp:,} (Precision: {tp/(tp+fp)*100:.1f}%)")
    print(f"    * False alarms from High (4-9 cases):   {fp_high:,} ({fp_high/fp*100 if fp>0 else 0:.1f}% of all FPs)")
    print(f"    * False alarms from Medium (1-3 cases): {fp_med:,}")
    print(f"    * False alarms from Low (0 cases):      {fp_low:,}")

    print(f"\nConfusion Matrix (Test Set):")
    print(pd.DataFrame(cm, index=[f"Actual_{c}" for c in RISK_CATEGORIES], columns=[f"Pred_{c}" for c in RISK_CATEGORIES]))
    print("\n")
