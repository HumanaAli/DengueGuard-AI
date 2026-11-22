import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.model_loader import get_engine, FORBIDDEN_COLS

engine = get_engine()
df = pd.read_csv("data/processed/model_dataset.csv")

print("=== 1. PREDICTION TESTS ACROSS WARDS & DATES ===")
test_rows = [
    # Baseline winter row
    ("CCC_Ward_01", "2025-01-13"),
    # Outbreak monsoon row
    ("DSCC_Ward_01", "2024-09-02"),
    ("DNCC_Ward_03", "2024-08-19"),
    ("GCC_Ward_01", "2023-08-14"),
    ("NCC_Ward_02", "2023-09-11"),
]

for ward, dt in test_rows:
    row = df[(df["ward_id"] == ward) & (df["date"] == dt)]
    if len(row) == 0:
        continue
    res_2w = engine.predict(row, horizon="2w")
    res_4w = engine.predict(row, horizon="4w")
    print(f"Ward: {ward:<12} | Date: {dt} | Split: {row.iloc[0]['split']}")
    print(f"  Actual 2w: {row.iloc[0]['target_2w']:4.1f} -> Pred 2w: {res_2w['predicted_cases'][0]:4.1f} ({res_2w['risk_tier'][0]:<8})")
    print(f"  Actual 4w: {row.iloc[0]['target_4w']:4.1f} -> Pred 4w: {res_4w['predicted_cases'][0]:4.1f} ({res_4w['risk_tier'][0]:<8})")

print("\n=== 2. RISK CATEGORY DIVERSITY CHECK (TEST SET) ===")
test_df = df[df["split"] == "test"]
preds_2w = engine.predict(test_df, horizon="2w")
preds_4w = engine.predict(test_df, horizon="4w")

print("2-Week Predicted Tiers on Test Set:")
print(pd.Series(preds_2w["risk_tier"]).value_counts())
print("\n4-Week Predicted Tiers on Test Set:")
print(pd.Series(preds_4w["risk_tier"]).value_counts())

# Non-negativity check
assert (preds_2w["predicted_cases"] >= 0).all(), "Negative prediction found in 2w!"
assert (preds_4w["predicted_cases"] >= 0).all(), "Negative prediction found in 4w!"
print("\nNon-negativity verification: PASSED (all >= 0.0)")

print("\n=== 3. ROBUSTNESS & EDGE CASE TESTS ===")
# Test unseen category handling (OrdinalEncoder handle_unknown='use_encoded_value', unknown_value=-1)
sample_unseen = test_df.head(1).copy()
sample_unseen["ward_id"] = "UNKNOWN_WARD_99"
sample_unseen["city"] = "UNKNOWN_CITY"
try:
    res_unseen = engine.predict(sample_unseen, horizon="2w")
    print(f"Unseen category handling: PASSED -> Output: {res_unseen['predicted_cases'][0]:.2f} cases ({res_unseen['risk_tier'][0]})")
except Exception as e:
    print(f"Unseen category handling: FAILED with error: {e}")

# Test missing feature validation
sample_missing = test_df.head(1).copy().drop(columns=["rainfall_mm"])
try:
    engine.predict(sample_missing, horizon="2w")
    print("Missing feature check: FAILED (did not raise error)")
except Exception as e:
    print(f"Missing feature check: PASSED (raised {type(e).__name__}: {str(e)[:70]}...)")
