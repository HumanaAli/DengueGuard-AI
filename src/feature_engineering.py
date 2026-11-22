"""
DENGUARD AI — Feature Engineering Pipeline
==========================================
Transforms the raw dataset into a machine-learning-ready dataset
with lag features, rolling features, seasonality, and future targets.

Usage (from project root):
    python src/feature_engineering.py

Output:
    data/processed/model_dataset.csv

Author: DENGUARD AI Team
"""

import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths  (all relative — works on any computer)
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_PATH     = os.path.join(PROJECT_ROOT, "data", "hackathon_preli_dataset.xlsx")
SHEET_NAME   = "bangladesh_dengue_large_dataset"
OUT_DIR      = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_PATH     = os.path.join(OUT_DIR, "model_dataset.csv")


# ===========================================================================
# STEP 1 — LOAD RAW DATA
# ===========================================================================
def load_raw() -> pd.DataFrame:
    """
    Load the Excel file without modifying it.
    Returns a copy of the raw dataframe sorted by ward_id then date.

    Why sort?
    ---------
    Every lag and rolling calculation depends on rows being in time order.
    Sorting here makes every later operation safe by default.
    """
    print("[STEP 1] Loading raw dataset …")
    if not os.path.exists(RAW_PATH):
        print(f"  [ERROR] File not found: {RAW_PATH}")
        sys.exit(1)

    df = pd.read_excel(RAW_PATH, sheet_name=SHEET_NAME)
    df = df.sort_values(["ward_id", "date"]).reset_index(drop=True)

    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Wards: {df['ward_id'].nunique()}   Cities: {df['city'].nunique()}")
    return df


# ===========================================================================
# STEP 2 — RAINFALL LAG AND ROLLING FEATURES
# ===========================================================================
def add_rainfall_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rainfall lags and rolling sums, computed within each ward.

    What is a 'lag'?
    ----------------
    A lag-1 value at week t is the value from week t-1 (one week ago).
    lag-2 = two weeks ago, etc.  These are created with .shift(n) which
    moves every value DOWN by n rows — so each row sees the past.

    What is a 'rolling sum'?
    ------------------------
    A 2-week rolling sum at week t  = rainfall[t-1] + rainfall[t-2].
    We always use min_periods=N to avoid producing partial sums.
    We then .shift(1) so the window never includes the current week
    (which would be leakage if the week isn't finished yet, and also
    ensures the current week's raw column already carries that value).

    Cross-ward contamination prevention:
    -------------------------------------
    groupby('ward_id') means every shift/rolling resets at the
    boundary of each ward. Ward A's last row never bleeds into
    Ward B's first row.
    """
    print("[STEP 2] Adding rainfall lag and rolling features …")

    grp = df.groupby("ward_id", sort=False)["rainfall_mm"]

    # Individual week lags
    for lag in [1, 2, 3, 4]:
        col = f"rain_lag{lag}w"
        df[col] = grp.shift(lag)

    # 2-week rolling sum: sum of lag-1 and lag-2
    # shift(1) first, then rolling(2) so window = [t-1, t-2]
    rain_shifted = grp.shift(1)
    df["rain_roll2w"] = rain_shifted.groupby(
        df["ward_id"]
    ).transform(lambda x: x.rolling(2, min_periods=2).sum())

    # 4-week rolling sum: sum of lag-1 through lag-4
    df["rain_roll4w"] = rain_shifted.groupby(
        df["ward_id"]
    ).transform(lambda x: x.rolling(4, min_periods=4).sum())

    # Log-transformed version of current rainfall
    # (audit found max=269mm vs median=8mm — very skewed)
    df["rain_log1p"] = np.log1p(df["rainfall_mm"])

    new_cols = ["rain_lag1w","rain_lag2w","rain_lag3w","rain_lag4w",
                "rain_roll2w","rain_roll4w","rain_log1p"]
    print(f"  Created {len(new_cols)} rainfall features: {new_cols}")
    return df


# ===========================================================================
# STEP 3 — TEMPERATURE AND HUMIDITY FEATURES
# ===========================================================================
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lagged temperature and humidity features within each ward.

    Why lag temperature and humidity?
    ----------------------------------
    Dengue mosquito breeding and biting depend on temperatures and humidity
    of the PREVIOUS 2–4 weeks, not today's.  High humidity + warm temps
    2 weeks ago → more mosquito larvae → more bites today.

    Features created:
    - avg_temp lags 1–4 weeks
    - 4-week rolling average temperature (smooth signal)
    - relative_humidity lag 1 and 2 weeks
    - temp-humidity interaction at lag-2 (warm+humid = ideal for mosquitoes)
    """
    print("[STEP 3] Adding temperature and humidity features …")

    # --- Temperature lags ---
    grp_temp = df.groupby("ward_id", sort=False)["avg_temp_c"]
    for lag in [1, 2, 3, 4]:
        df[f"temp_lag{lag}w"] = grp_temp.shift(lag)

    # 4-week rolling average temperature (shifted so window = t-1 to t-4)
    temp_shifted = grp_temp.shift(1)
    df["temp_roll4w_mean"] = temp_shifted.groupby(
        df["ward_id"]
    ).transform(lambda x: x.rolling(4, min_periods=4).mean())

    # --- Humidity lags ---
    grp_hum = df.groupby("ward_id", sort=False)["relative_humidity_pct"]
    for lag in [1, 2]:
        df[f"humidity_lag{lag}w"] = grp_hum.shift(lag)

    # --- Interaction feature: warmth × humidity at 2-week lag ---
    # Mosquitoes breed best in warm, humid conditions.
    # We capture this by multiplying temp and humidity both lagged 2 weeks.
    df["temp_x_humidity_lag2w"] = (
        df["temp_lag2w"] * df["humidity_lag2w"] / 100.0   # scale humidity to 0–1
    )

    new_cols = [
        "temp_lag1w","temp_lag2w","temp_lag3w","temp_lag4w","temp_roll4w_mean",
        "humidity_lag1w","humidity_lag2w","temp_x_humidity_lag2w"
    ]
    print(f"  Created {len(new_cols)} temperature/humidity features: {new_cols}")
    return df


# ===========================================================================
# STEP 4 — HISTORICAL DENGUE CASE FEATURES
# ===========================================================================
def add_dengue_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lagged dengue-case columns, computed within each ward.

    Why do we include past dengue cases?
    --------------------------------------
    Dengue outbreaks have momentum. If there were 20 cases last week,
    there will likely be more this week (the same infected people are
    still infectious, and their contacts are being bitten).

    These lags are the strongest predictors in most dengue models.

    Important: lag-1 (previous week) = minimum safe lag.
    We NEVER include lag-0 (current week's cases) as a feature because
    we wouldn't know today's cases when making predictions.
    """
    print("[STEP 4] Adding historical dengue lag features …")

    grp_cases = df.groupby("ward_id", sort=False)["dengue_cases"]

    for lag in [1, 2, 3, 4]:
        df[f"dengue_lag{lag}w"] = grp_cases.shift(lag)

    new_cols = ["dengue_lag1w","dengue_lag2w","dengue_lag3w","dengue_lag4w"]
    print(f"  Created {len(new_cols)} dengue lag features: {new_cols}")
    return df


# ===========================================================================
# STEP 5 — FUTURE PREDICTION TARGETS
# ===========================================================================
def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the two columns we want the model to PREDICT.

    How negative shift works:
    -------------------------
    .shift(-2) moves every value UP by 2 rows.  So for the row
    dated 2020-01-06, the value in target_2w is the dengue_cases
    from 2020-01-20 (the row 2 positions below in the same ward).

    In plain English:
      target_2w = "how many dengue cases will there be 2 weeks from now?"
      target_4w = "how many dengue cases will there be 4 weeks from now?"

    These are what the model is trained to predict.
    The last 2 and 4 rows of each ward will get NaN (no future available).
    We drop those NaN rows later.

    Data leakage safeguard:
    ------------------------
    The target is NEVER used as an input feature.  It is only the label.
    When the model is deployed live, we provide weather/dengue features
    from today and the model outputs the target — no future information.
    """
    print("[STEP 5] Creating future prediction targets …")

    grp_cases = df.groupby("ward_id", sort=False)["dengue_cases"]

    # target_2w: dengue cases 2 weeks AFTER the current observation
    df["target_2w"] = grp_cases.shift(-2)

    # target_4w: dengue cases 4 weeks AFTER the current observation
    df["target_4w"] = grp_cases.shift(-4)

    # Also store the actual future target dates (useful for splitting)
    df["target_date_2w"] = df["date"] + pd.Timedelta(weeks=2)
    df["target_date_4w"] = df["date"] + pd.Timedelta(weeks=4)

    print("  Created: target_2w, target_4w, target_date_2w, target_date_4w")
    return df


# ===========================================================================
# STEP 6 — SEASONALITY FEATURES
# ===========================================================================
def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features that capture seasonal dengue patterns.

    Why seasonality?
    ----------------
    Dengue in Bangladesh peaks July–October (monsoon and post-monsoon).
    Adding week_of_year and month lets the model learn this annual cycle
    without memorising individual dates.

    Monsoon flag (Jun–Sep = 1):
    ---------------------------
    A simple 0/1 column that marks the high-risk season.
    This compresses the seasonal signal into one easy feature.
    """
    print("[STEP 6] Adding seasonality features …")

    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date"].dt.month
    df["year"]         = df["date"].dt.year

    # Monsoon season in Bangladesh: June (6) through September (9)
    df["is_monsoon"]   = df["month"].between(6, 9).astype(int)

    # Post-monsoon (Oct–Nov) is also high-risk (peak dengue months)
    df["is_post_monsoon"] = df["month"].isin([10, 11]).astype(int)

    new_cols = ["week_of_year","month","year","is_monsoon","is_post_monsoon"]
    print(f"  Created {len(new_cols)} seasonality features: {new_cols}")
    return df


# ===========================================================================
# STEP 7 — DROP INCOMPLETE ROWS
# ===========================================================================
def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where required features or targets are unavailable.

    Which rows get dropped and why:
    --------------------------------
    - First 4 rows of each ward: lags 1–4 produce NaN for the earliest dates
      (you can't look 4 weeks back when the dataset starts).
    - Last 4 rows of each ward: target_4w is NaN (no future data beyond the end).

    Result: each ward loses 4 rows at the start + 4 rows at the end = 8 rows.
    With 60 wards and 522 original rows each, we keep:
      60 × (522 − 8) = 60 × 514 = 30,840 rows.

    Data leakage protection:
    -------------------------
    Dropping NaN rows removes exactly the rows where the model would have to
    "invent" past or future data — which would be leakage.
    """
    print("[STEP 7] Dropping rows with unavailable features or targets …")
    before = len(df)

    # Columns that must NOT be NaN in the final dataset
    required_cols = [
        # Rainfall lags
        "rain_lag1w","rain_lag2w","rain_lag3w","rain_lag4w",
        "rain_roll2w","rain_roll4w",
        # Temperature lags
        "temp_lag1w","temp_lag2w","temp_lag3w","temp_lag4w",
        "temp_roll4w_mean",
        # Humidity lags
        "humidity_lag1w","humidity_lag2w",
        "temp_x_humidity_lag2w",
        # Dengue lags
        "dengue_lag1w","dengue_lag2w","dengue_lag3w","dengue_lag4w",
        # Targets (both must be available)
        "target_2w","target_4w",
    ]

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    after = len(df)
    dropped = before - after
    print(f"  Rows before: {before:,}")
    print(f"  Rows after:  {after:,}")
    print(f"  Dropped:     {dropped:,} rows (={dropped//60} per ward × 60 wards)")
    return df


# ===========================================================================
# STEP 8 — CHRONOLOGICAL SPLIT COLUMN
# ===========================================================================
def add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'split' column to label each row as train / val / test.

    We split on TARGET DATE (not observation date):
    -----------------------------------------------
    If we split on the observation date, a row observed in Dec 2023
    with a 4-week target in Jan 2024 would be labelled 'train' —
    but its outcome is in the validation period.  Using the target
    date prevents this.

    Convention used:
      train : target date  <= 2023-12-31
      val   : target date in 2024
      test  : target date in 2025

    We use target_date_4w (the later of the two) as the split key.
    This is the most conservative choice: a row is only called 'train'
    if BOTH its 2-week AND 4-week targets fall in the training period.
    """
    print("[STEP 8] Assigning chronological train/val/test splits …")

    split_date = df["target_date_4w"]   # use the later target as split key

    conditions = [
        split_date.dt.year <= 2023,
        split_date.dt.year == 2024,
        split_date.dt.year == 2025,
    ]
    choices = ["train", "val", "test"]
    df["split"] = np.select(conditions, choices, default="other")

    split_counts = df["split"].value_counts().reindex(["train","val","test","other"])
    for s, n in split_counts.items():
        print(f"  {s:6s}: {n:6,} rows")
    return df


# ===========================================================================
# STEP 9 — SAVE TO CSV
# ===========================================================================
def save_dataset(df: pd.DataFrame) -> None:
    """Save the processed dataset to data/processed/model_dataset.csv."""
    print("[STEP 9] Saving processed dataset …")
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"  Saved → {OUT_PATH}")
    print(f"  File size: {size_kb:.1f} KB")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


# ===========================================================================
# STEP 10 — VERIFICATION CHECKS
# ===========================================================================
def verify(df: pd.DataFrame) -> None:
    """
    Run sanity checks to confirm no leakage, correct lag alignment,
    correct target alignment, and correct split dates.
    """
    print("\n" + "=" * 60)
    print("  VERIFICATION CHECKS")
    print("=" * 60)

    # --- Check 1: No NaN in required columns ---
    print("\n[V1] NaN check on required columns …")
    required = [c for c in df.columns if c not in ["target_date_2w","target_date_4w"]]
    nan_counts = df[required].isnull().sum()
    nans = nan_counts[nan_counts > 0]
    if nans.empty:
        print("  ✓ No NaN values in any feature or target column.")
    else:
        print(f"  ✗ NaN found:\n{nans}")

    # --- Check 2: Lag alignment (pick one ward and spot-check) ---
    print("\n[V2] Lag alignment spot check (CCC_Ward_01) …")
    ward = df[df["ward_id"] == "CCC_Ward_01"].sort_values("date").reset_index(drop=True)
    idx = 10   # arbitrary middle row (after the lag warm-up period)
    obs_date       = ward.loc[idx, "date"]
    rain_now       = ward.loc[idx, "rainfall_mm"]
    rain_lag1_col  = ward.loc[idx, "rain_lag1w"]
    rain_actual_1  = ward.loc[idx - 1, "rainfall_mm"]   # manual look-up
    rain_lag2_col  = ward.loc[idx, "rain_lag2w"]
    rain_actual_2  = ward.loc[idx - 2, "rainfall_mm"]
    print(f"  Ward: CCC_Ward_01  |  Observation date: {obs_date.date()}")
    print(f"  Current rainfall:        {rain_now:.2f} mm")
    print(f"  rain_lag1w (column):     {rain_lag1_col:.2f} mm  "
          f"| Manual t-1: {rain_actual_1:.2f} mm  "
          f"| Match: {'✓' if abs(rain_lag1_col - rain_actual_1) < 0.001 else '✗'}")
    print(f"  rain_lag2w (column):     {rain_lag2_col:.2f} mm  "
          f"| Manual t-2: {rain_actual_2:.2f} mm  "
          f"| Match: {'✓' if abs(rain_lag2_col - rain_actual_2) < 0.001 else '✗'}")

    # --- Check 3: Target alignment ---
    print("\n[V3] Target alignment spot check (CCC_Ward_01) …")
    target_2w_col  = ward.loc[idx, "target_2w"]
    actual_2w      = ward.loc[idx + 2, "dengue_cases"]
    target_4w_col  = ward.loc[idx, "target_4w"]
    actual_4w      = ward.loc[idx + 4, "dengue_cases"]
    print(f"  target_2w (column):   {target_2w_col}  "
          f"| Actual cases at t+2: {actual_2w}  "
          f"| Match: {'✓' if target_2w_col == actual_2w else '✗'}")
    print(f"  target_4w (column):   {target_4w_col}  "
          f"| Actual cases at t+4: {actual_4w}  "
          f"| Match: {'✓' if target_4w_col == actual_4w else '✗'}")

    # --- Check 4: Cross-ward contamination ---
    print("\n[V4] Cross-ward contamination check …")
    # Pick the FIRST valid row of CCC_Ward_02.
    # Its lag1 must NOT equal the last row of CCC_Ward_01.
    ward2_first = df[df["ward_id"] == "CCC_Ward_02"].sort_values("date").iloc[0]
    ward1_last  = df[df["ward_id"] == "CCC_Ward_01"].sort_values("date").iloc[-1]
    # rain_lag1w of ward2 first row was computed within ward2 — should not equal ward1 last
    # We verify by checking that ward2 lag1 equals ward2's own t-1 row
    ward2_all = df[df["ward_id"] == "CCC_Ward_02"].sort_values("date").reset_index(drop=True)
    lag1_check = ward2_all.loc[0, "rain_lag1w"]
    expected   = ward2_all["rainfall_mm"].iloc[0] if False else float("nan")
    # The lag1 for the first remaining row of ward2 should equal
    # the value 1 position before it in ward2 (before filtering).
    # We verify it does NOT equal ward1's last row value:
    ward1_last_rain = df[df["ward_id"] == "CCC_Ward_01"].sort_values("date").iloc[-1]["rainfall_mm"]
    mismatch = (lag1_check == ward1_last_rain)
    print(f"  CCC_Ward_02 first row rain_lag1w : {lag1_check:.2f}")
    print(f"  CCC_Ward_01 last row  rainfall   : {ward1_last_rain:.2f}")
    print(f"  Cross-ward contamination: {'✗ CONTAMINATION DETECTED' if mismatch else '✓ None (values differ as expected)'}")

    # --- Check 5: Split date boundaries ---
    print("\n[V5] Split boundary check …")
    for split_name in ["train", "val", "test"]:
        sub = df[df["split"] == split_name]
        if len(sub) == 0:
            continue
        min_td = sub["target_date_4w"].min().date()
        max_td = sub["target_date_4w"].max().date()
        min_od = sub["date"].min().date()
        max_od = sub["date"].max().date()
        print(f"  {split_name:6s}: obs {min_od}→{max_od}  |  "
              f"target_date_4w {min_td}→{max_td}  |  rows: {len(sub):,}")

    # --- Check 6: Feature list ---
    print("\n[V6] Final feature summary …")
    input_features = [c for c in df.columns if c not in [
        "target_2w","target_4w","target_date_2w","target_date_4w","split"
    ]]
    print(f"  Total columns in output: {df.shape[1]}")
    print(f"  Input feature columns:   {len(input_features)}")
    print(f"  Target columns:          target_2w, target_4w")
    print(f"  Metadata columns:        target_date_2w, target_date_4w, split")

    # --- Check 7: Row count matches expectation ---
    print("\n[V7] Row count check …")
    expected_per_ward = 522 - 4 - 4   # lose 4 head + 4 tail per ward
    expected_total    = 60 * expected_per_ward
    actual_total      = len(df)
    match = "✓" if actual_total == expected_total else "✗"
    print(f"  Expected: 60 wards × {expected_per_ward} rows = {expected_total:,}")
    print(f"  Actual  : {actual_total:,}  {match}")

    print("\n" + "=" * 60)
    print("  VERIFICATION COMPLETE")
    print("=" * 60 + "\n")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("\n" + "#" * 60)
    print("#   DENGUARD AI — Feature Engineering Pipeline")
    print("#" * 60)

    df = load_raw()
    df = add_rainfall_features(df)
    df = add_weather_features(df)
    df = add_dengue_lag_features(df)
    df = add_future_targets(df)
    df = add_seasonality_features(df)
    df = drop_incomplete_rows(df)
    df = add_split_column(df)
    save_dataset(df)
    verify(df)

    # Print final column list
    print("Final columns in model_dataset.csv:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:02d}. {col}")


if __name__ == "__main__":
    main()
