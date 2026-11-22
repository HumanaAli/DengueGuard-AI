import hashlib
import json
import joblib
import pandas as pd
import numpy as np

# 1. Dataset & Pipeline
raw = pd.read_excel("data/hackathon_preli_dataset.xlsx", sheet_name="bangladesh_dengue_large_dataset")
processed = pd.read_csv("data/processed/model_dataset.csv")

print("=== 1. DATASET PROPERTIES ===")
print(f"Raw shape:       {raw.shape} (Rows: {raw.shape[0]:,}, Cols: {raw.shape[1]})")
print(f"Processed shape: {processed.shape} (Rows: {processed.shape[0]:,}, Cols: {processed.shape[1]})")
print(f"Raw date range:  {raw['date'].min()} to {raw['date'].max()} (Total unique dates: {raw['date'].nunique()})")
print(f"Proc date range: {processed['date'].min()} to {processed['date'].max()} (Total unique dates: {processed['date'].nunique()})")
print(f"Raw Wards:       {raw['ward_id'].nunique()} across {raw['city'].nunique()} cities")
print(f"Proc Wards:      {processed['ward_id'].nunique()} across {processed['city'].nunique()} cities")
print(f"Missing values:  Raw={raw.isnull().sum().sum()}, Proc={processed.isnull().sum().sum()}")

# 2. Target Shifts
raw_sorted = raw.sort_values(["ward_id", "date"]).reset_index(drop=True)
proc_sorted = processed.sort_values(["ward_id", "date"]).reset_index(drop=True)

# Spot check across multiple wards
shifts_valid = True
for ward in ["CCC_Ward_01", "DNCC_Ward_05", "GCC_Ward_02"]:
    r_w = raw_sorted[raw_sorted["ward_id"] == ward].reset_index(drop=True)
    p_w = proc_sorted[proc_sorted["ward_id"] == ward].reset_index(drop=True)
    for p_idx in [0, 50, 150, 300, 500]:
        p_date = p_w.loc[p_idx, "date"]
        r_idx = r_w[r_w["date"] == p_date].index[0]
        t2_act = p_w.loc[p_idx, "target_2w"]
        t2_exp = r_w.loc[r_idx + 2, "dengue_cases"]
        t4_act = p_w.loc[p_idx, "target_4w"]
        t4_exp = r_w.loc[r_idx + 4, "dengue_cases"]
        if t2_act != t2_exp or t4_act != t4_exp:
            shifts_valid = False
            print(f"Mismatch in {ward} at date {p_date}!")
print(f"Target shifts verified against raw data: {shifts_valid}")

# 3. Target dates check
td2_match = (pd.to_datetime(processed["target_date_2w"]) == pd.to_datetime(processed["date"]) + pd.Timedelta(weeks=2)).all()
td4_match = (pd.to_datetime(processed["target_date_4w"]) == pd.to_datetime(processed["date"]) + pd.Timedelta(weeks=4)).all()
print(f"Target date 2w = date + 2 weeks: {td2_match}")
print(f"Target date 4w = date + 4 weeks: {td4_match}")

# 4. Split check
for s in ["train", "val", "test"]:
    sub = processed[processed["split"] == s]
    d_min, d_max = sub["date"].min(), sub["date"].max()
    t4_min, t4_max = sub["target_date_4w"].min(), sub["target_date_4w"].max()
    print(f"Split {s:<6}: obs [{d_min} to {d_max}] | target_date_4w [{t4_min} to {t4_max}] | N={len(sub):,}")
