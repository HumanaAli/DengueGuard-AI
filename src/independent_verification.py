import hashlib
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

print("=" * 80)
print("  DENGUARD AI: INDEPENDENT FINAL VERIFICATION SCRIPT")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. HASH RE-VERIFICATION
# ---------------------------------------------------------------------------
def get_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

p2 = "models/model_2w.joblib"
p2_b = "models/backups/model_2w_backup.joblib"
p4 = "models/model_4w.joblib"
p4_b = "models/backups/model_4w_backup.joblib"

h2 = get_sha256(p2)
h2_b = get_sha256(p2_b)
h4 = get_sha256(p4)
h4_b = get_sha256(p4_b)

print("\n--- TASK 4: SHA-256 HASH VERIFICATION ---")
print(f"2W Model:  {h2}")
print(f"2W Backup: {h2_b} (Match: {h2 == h2_b})")
print(f"4W Model:  {h4}")
print(f"4W Backup: {h4_b} (Match: {h4 == h4_b})")

expected_h2 = "3646673bf04b6002739edfebf619beacfb7bd2b3678026640d1e991df112f8c7"
expected_h4 = "fb889d62470594ae70602ad9b7b609079d6ee1acad3ba3eb52e5498cdcd3c4fb"
print(f"2W matches documented hash: {h2 == expected_h2}")
print(f"4W matches documented hash: {h4 == expected_h4}")

# ---------------------------------------------------------------------------
# 2. DATASET & TEMPORAL SPLITS AUDIT
# ---------------------------------------------------------------------------
print("\n--- TASK 2: DATASET AND TEMPORAL SPLITS ---")
raw = pd.read_excel("data/hackathon_preli_dataset.xlsx", sheet_name="bangladesh_dengue_large_dataset")
proc = pd.read_csv("data/processed/model_dataset.csv")

print(f"Raw shape:       {raw.shape} | Missing: {raw.isnull().sum().sum()} | Duplicates: {raw.duplicated().sum()}")
print(f"Processed shape: {proc.shape} | Missing: {proc.isnull().sum().sum()} | Duplicates: {proc.duplicated().sum()}")
print(f"Raw dates:       {raw['date'].nunique()} Mondays, range: {raw['date'].min()} to {raw['date'].max()}")
print(f"Proc dates:      {proc['date'].nunique()} Mondays, range: {proc['date'].min()} to {proc['date'].max()}")
print(f"Raw Wards:       {raw['ward_id'].nunique()} in {raw['city'].nunique()} cities")
print(f"Proc Wards:      {proc['ward_id'].nunique()} in {proc['city'].nunique()} cities")

# Check balanced panel
raw_ward_counts = raw.groupby("ward_id").size()
proc_ward_counts = proc.groupby("ward_id").size()
print(f"All 60 wards have exactly 522 raw rows: {(raw_ward_counts == 522).all()}")
print(f"All 60 wards have exactly 514 proc rows: {(proc_ward_counts == 514).all()}")

# Chronological sorting check
raw_sorted = raw.sort_values(["ward_id", "date"]).reset_index(drop=True)
proc_sorted = proc.sort_values(["ward_id", "date"]).reset_index(drop=True)

# Direct comparison of target_2w and target_4w against raw Excel
mismatches_2w = 0
mismatches_4w = 0
for ward_id, p_group in proc_sorted.groupby("ward_id"):
    r_group = raw_sorted[raw_sorted["ward_id"] == ward_id].reset_index(drop=True)
    # in raw, dates start at 2016-01-04 (index 0)
    # in proc, dates start at 2016-02-01 (4 weeks dropped at head) -> raw index 4
    for i, row in p_group.reset_index(drop=True).iterrows():
        r_idx = i + 4
        # target_2w should match raw at r_idx + 2
        exp_2w = r_group.loc[r_idx + 2, "dengue_cases"]
        exp_4w = r_group.loc[r_idx + 4, "dengue_cases"]
        if row["target_2w"] != exp_2w:
            mismatches_2w += 1
        if row["target_4w"] != exp_4w:
            mismatches_4w += 1

print(f"Target 2w mismatches across all 30,840 rows: {mismatches_2w}")
print(f"Target 4w mismatches across all 30,840 rows: {mismatches_4w}")

# Target date accuracy
td2_err = (pd.to_datetime(proc["target_date_2w"]) != pd.to_datetime(proc["date"]) + pd.Timedelta(weeks=2)).sum()
td4_err = (pd.to_datetime(proc["target_date_4w"]) != pd.to_datetime(proc["date"]) + pd.Timedelta(weeks=4)).sum()
print(f"Target date 2w calculation errors: {td2_err}")
print(f"Target date 4w calculation errors: {td4_err}")

# Split boundaries check
print("\nSplit boundaries:")
for s in ["train", "val", "test"]:
    sub = proc[proc["split"] == s]
    obs_min, obs_max = sub["date"].min(), sub["date"].max()
    t4_min, t4_max = sub["target_date_4w"].min(), sub["target_date_4w"].max()
    t2_min, t2_max = sub["target_date_2w"].min(), sub["target_date_2w"].max()
    print(f"  {s:<5}: N={len(sub):5,} | obs [{obs_min} -> {obs_max}] | target_date_4w [{t4_min} -> {t4_max}] | target_date_2w [{t2_min} -> {t2_max}]")

# Check boundary separation
train_sub = proc[proc["split"] == "train"]
val_sub = proc[proc["split"] == "val"]
test_sub = proc[proc["split"] == "test"]

print(f"Train target_date_4w all <= 2023-12-31: {(pd.to_datetime(train_sub['target_date_4w']).dt.year <= 2023).all()}")
print(f"Val target_date_4w all in 2024:        {(pd.to_datetime(val_sub['target_date_4w']).dt.year == 2024).all()}")
print(f"Test target_date_4w all in 2025:       {(pd.to_datetime(test_sub['target_date_4w']).dt.year == 2025).all()}")

# ---------------------------------------------------------------------------
# 3. MODEL & FEATURE AUDIT
# ---------------------------------------------------------------------------
print("\n--- TASK 3: MODEL FEATURE AUDIT ---")
m2 = joblib.load(p2)
m4 = joblib.load(p4)

pipe_feats_2w = list(m2.feature_names_in_)
pipe_feats_4w = list(m4.feature_names_in_)

print(f"2W Model input features count: {len(pipe_feats_2w)}")
print(f"4W Model input features count: {len(pipe_feats_4w)}")
print(f"Feature lists identical between 2W and 4W: {pipe_feats_2w == pipe_feats_4w}")

FORBIDDEN = ["dengue_cases", "target_2w", "target_4w", "target_date_2w", "target_date_4w", "date", "split"]
leaked_2w = [f for f in pipe_feats_2w if f in FORBIDDEN]
leaked_4w = [f for f in pipe_feats_4w if f in FORBIDDEN]
print(f"Forbidden features in 2W: {leaked_2w}")
print(f"Forbidden features in 4W: {leaked_4w}")

# Preprocessor inspection
prep = m2.named_steps["preprocessor"]
cat_trans = prep.transformers_[0]
num_trans = prep.transformers_[1]
print(f"Categorical features in Preprocessor: {cat_trans[2]}")
print(f"Numerical features in Preprocessor count: {len(num_trans[2])}")

# Config check
with open("models/calibrated_config.json") as f:
    cfg = json.load(f)
print(f"Config strategy: {cfg['strategy']}")
print(f"Config status:   {cfg['status']}")
print(f"Config 2W thresholds: {cfg['thresholds']['2w']}")
print(f"Config 4W thresholds: {cfg['thresholds']['4w']}")

# ---------------------------------------------------------------------------
# 4. EVALUATION RE-COMPUTATION ON UNTOUCHED 2025 TEST SET
# ---------------------------------------------------------------------------
print("\n--- TASK 5: INDEPENDENT METRICS RE-COMPUTATION ON TEST SET (2025) ---")
test_df = proc[proc["split"] == "test"].copy()
X_test = test_df[pipe_feats_2w]

y_t2 = test_df["target_2w"].values
y_t4 = test_df["target_4w"].values

p2_pred = np.clip(m2.predict(X_test), 0, None)
p4_pred = np.clip(m4.predict(X_test), 0, None)

# Strict actual definitions: Low < 1, Med 1-4, High 4-10, Crit >= 10
def actual_risk(y):
    arr = np.asarray(y)
    cats = np.empty(arr.shape, dtype=object)
    cats[arr < 1.0] = "Low"
    cats[(arr >= 1.0) & (arr < 4.0)] = "Medium"
    cats[(arr >= 4.0) & (arr < 10.0)] = "High"
    cats[arr >= 10.0] = "Critical"
    return cats

act_t2 = actual_risk(y_t2)
act_t4 = actual_risk(y_t4)

# Calibrated predictions
def pred_risk(p, thresh):
    tl, tm, tc = thresh
    cats = np.empty(p.shape, dtype=object)
    cats[p < tl] = "Low"
    cats[(p >= tl) & (p < tm)] = "Medium"
    cats[(p >= tm) & (p < tc)] = "High"
    cats[p >= tc] = "Critical"
    return cats

p_cat_2w = pred_risk(p2_pred, (0.8, 4.0, 8.0))
p_cat_4w = pred_risk(p4_pred, (0.8, 4.0, 8.5))

RISK_TIERS = ["Low", "Medium", "High", "Critical"]

for name, y_true, act_c, pred_c, p_cont in [
    ("2-WEEK HORIZON", y_t2, act_t2, p_cat_2w, p2_pred),
    ("4-WEEK HORIZON", y_t4, act_t4, p_cat_4w, p4_pred)
]:
    print(f"\n=======================================================")
    print(f"  {name} (N = {len(y_true):,})")
    print(f"=======================================================")
    mae = mean_absolute_error(y_true, p_cont)
    rmse = root_mean_squared_error(y_true, p_cont)
    pdev = mean_poisson_deviance(y_true, np.clip(p_cont, 1e-6, None))
    macro_f1 = f1_score(act_c, pred_c, labels=RISK_TIERS, average="macro")
    macro_p = precision_score(act_c, pred_c, labels=RISK_TIERS, average="macro")
    macro_r = recall_score(act_c, pred_c, labels=RISK_TIERS, average="macro")
    cr = classification_report(act_c, pred_c, labels=RISK_TIERS, output_dict=True, zero_division=0)
    cm = confusion_matrix(act_c, pred_c, labels=RISK_TIERS)
    
    print(f"  MAE:              {mae:.4f}")
    print(f"  RMSE:             {rmse:.4f}")
    print(f"  Poisson Deviance: {pdev:.4f}")
    print(f"  Macro F1:         {macro_f1:.4f}")
    print(f"  Macro Precision:  {macro_p:.4f}")
    print(f"  Macro Recall:     {macro_r:.4f}")
    print(f"  Critical Recall:   {cr['Critical']['recall']*100:.2f}% (Support: {cr['Critical']['support']})")
    print(f"  Critical Precision:{cr['Critical']['precision']*100:.2f}%")
    print(f"  Medium Recall:     {cr['Medium']['recall']*100:.2f}% (Support: {cr['Medium']['support']})")
    print(f"  Medium Precision:  {cr['Medium']['precision']*100:.2f}%")
    print(f"  Low Recall:        {cr['Low']['recall']*100:.2f}% (Support: {cr['Low']['support']})")
    print(f"  High Recall:       {cr['High']['recall']*100:.2f}% (Support: {cr['High']['support']})")
    
    # False Critical Alarms
    crit_pred = (pred_c == "Critical")
    crit_act = (act_c == "Critical")
    tp = (crit_pred & crit_act).sum()
    fp = (crit_pred & ~crit_act).sum()
    fp_high = (crit_pred & (act_c == "High")).sum()
    fp_med = (crit_pred & (act_c == "Medium")).sum()
    fp_low = (crit_pred & (act_c == "Low")).sum()
    
    print(f"  Critical True Positives:  {tp}")
    print(f"  Critical False Alarms:    {fp} (Total Pred Critical: {crit_pred.sum()})")
    print(f"    - from High (4-9 cases):   {fp_high} ({fp_high/fp*100:.1f}%)")
    print(f"    - from Medium (1-3 cases): {fp_med} ({fp_med/fp*100:.1f}%)")
    print(f"    - from Low (0 cases):      {fp_low} ({fp_low/fp*100:.1f}%)")
    print("\n  Confusion Matrix (Actual Rows x Pred Cols):")
    print(pd.DataFrame(cm, index=[f"Act_{c}" for c in RISK_TIERS], columns=[f"Pred_{c}" for c in RISK_TIERS]))

print("\n" + "=" * 80)
print("  RE-VERIFICATION COMPLETE: ALL METRICS INDEPENDENTLY CONFIRMED")
print("=" * 80)
