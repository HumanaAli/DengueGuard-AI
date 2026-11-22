"""
DENGUARD AI — Data Audit Script
================================
Reusable script that performs a thorough audit of the
hackathon_preli_dataset.xlsx file.

Run from the project root:
    python src/data_audit.py

Or from inside the src/ folder:
    python data_audit.py

All file paths are relative so the project works on any computer.
"""

import os
import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. Locate the dataset (works from project root OR from src/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)          # one level up from src/
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "hackathon_preli_dataset.xlsx")
SHEET_NAME = "bangladesh_dengue_large_dataset"


def separator(title: str) -> None:
    """Print a visible section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_data() -> pd.DataFrame:
    """Load the Excel dataset and return a DataFrame."""
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at: {DATA_PATH}")
        sys.exit(1)

    print(f"[INFO] Loading dataset from: {DATA_PATH}")
    xf = pd.ExcelFile(DATA_PATH)
    print(f"[INFO] Sheets found: {xf.sheet_names}")
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    print(f"[INFO] Loaded sheet '{SHEET_NAME}'")
    return df


def check_shape(df: pd.DataFrame) -> None:
    separator("1. Dataset Shape")
    rows, cols = df.shape
    print(f"  Rows    : {rows:,}")
    print(f"  Columns : {cols}")


def check_columns(df: pd.DataFrame) -> None:
    separator("2. Column Names and Data Types")
    for col in df.columns:
        print(f"  {col:<30} dtype={str(df[col].dtype):<18} "
              f"nulls={df[col].isnull().sum():<5} unique={df[col].nunique()}")


def check_date_range(df: pd.DataFrame) -> None:
    separator("3. Date Range and Weekly Frequency")
    dates = df["date"].sort_values().unique()
    print(f"  Earliest date : {df['date'].min().date()}")
    print(f"  Latest date   : {df['date'].max().date()}")
    print(f"  Unique dates  : {len(dates)}")

    gaps = pd.Series(dates).diff().dropna()
    gap_counts = gaps.dt.days.value_counts()
    print("  Gap sizes (days) → count:")
    for gap, cnt in gap_counts.items():
        print(f"    {int(gap)} days → {cnt} occurrences")

    # Expected: 60 wards × 522 dates = 31,320 rows
    expected_rows = df["ward_id"].nunique() * len(dates)
    print(f"  Expected rows (wards × dates) : {expected_rows:,}")
    print(f"  Actual rows                   : {len(df):,}")
    match = "✓ Matches" if expected_rows == len(df) else "✗ MISMATCH"
    print(f"  Balance check                 : {match}")


def check_cities_wards(df: pd.DataFrame) -> None:
    separator("4. Cities and Wards")
    print(f"  Total unique cities : {df['city'].nunique()}")
    print(f"  Total unique wards  : {df['ward_id'].nunique()}")
    city_ward = df.groupby("city")["ward_id"].nunique()
    print("  Wards per city:")
    for city, count in city_ward.items():
        print(f"    {city:<20} {count} wards")


def check_missing(df: pd.DataFrame) -> None:
    separator("5. Missing Values")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found in any column. ✓")
    else:
        print(missing[missing > 0])


def check_duplicates(df: pd.DataFrame) -> None:
    separator("6. Duplicate Records")
    exact_dups = df.duplicated().sum()
    key_dups = df.duplicated(subset=["city", "ward_id", "date"]).sum()
    print(f"  Exact duplicate rows              : {exact_dups}")
    print(f"  Duplicate city+ward+date combos   : {key_dups}")
    if exact_dups == 0 and key_dups == 0:
        print("  No duplicates found. ✓")


def check_dengue_distribution(df: pd.DataFrame) -> None:
    separator("7. Dengue Cases Distribution")
    col = df["dengue_cases"]
    zero_rows = (col == 0).sum()
    nonzero_rows = (col > 0).sum()
    print(f"  Total cases (sum)     : {col.sum():,}")
    print(f"  Zero-case rows        : {zero_rows:,}  ({100*zero_rows/len(df):.1f}%)")
    print(f"  Non-zero rows         : {nonzero_rows:,}  ({100*nonzero_rows/len(df):.1f}%)")
    print(f"  Min                   : {col.min()}")
    print(f"  Max                   : {col.max()}")
    print(f"  Mean                  : {col.mean():.3f}")
    print(f"  Median                : {col.median():.1f}")
    print(f"  Std deviation         : {col.std():.3f}")
    print(f"  Distinct values       : {col.nunique()}")
    print("  Top 10 value counts:")
    for val, cnt in col.value_counts().head(10).items():
        print(f"    {val:>3} cases → {cnt:>5} rows")

    # Per-city totals
    print("\n  Dengue cases per city:")
    city_stats = df.groupby("city")["dengue_cases"].agg(
        total="sum", mean="mean", max="max", std="std"
    ).round(2)
    print(city_stats.to_string())


def check_ward_vulnerability(df: pd.DataFrame) -> None:
    separator("8. Ward Vulnerability Features")
    vuln_cols = ["population_density_sqkm", "water_logging_risk_index", "sanitation_score"]
    ward_meta = df[["ward_id", "city"] + vuln_cols].drop_duplicates()
    print(f"  Unique ward metadata rows : {len(ward_meta)}")

    # Check if static per ward
    var_per_ward = df.groupby("ward_id")[vuln_cols].nunique()
    any_varies = (var_per_ward > 1).any().any()
    print(f"  Features static per ward  : {'YES ✓' if not any_varies else 'NO — features vary!'}")

    print("\n  Feature ranges across all wards:")
    for col in vuln_cols:
        print(f"    {col:<35} min={ward_meta[col].min()}  max={ward_meta[col].max()}")

    print("\n  Sample ward metadata (first 5):")
    print(ward_meta.head(5).to_string(index=False))


def check_weather_sanity(df: pd.DataFrame) -> None:
    separator("9. Weather Sanity Checks")
    bad_temp = df[(df["avg_temp_c"] < df["min_temp_c"]) |
                  (df["avg_temp_c"] > df["max_temp_c"])]
    neg_rain = (df["rainfall_mm"] < 0).sum()
    bad_hum = ((df["relative_humidity_pct"] < 0) |
               (df["relative_humidity_pct"] > 100)).sum()
    neg_cases = (df["dengue_cases"] < 0).sum()

    print(f"  Rows where avg_temp not between min/max : {len(bad_temp)}")
    print(f"  Rows with negative rainfall             : {neg_rain}")
    print(f"  Rows with humidity outside 0–100        : {bad_hum}")
    print(f"  Rows with negative dengue_cases         : {neg_cases}")

    if len(bad_temp) == 0 and neg_rain == 0 and bad_hum == 0 and neg_cases == 0:
        print("  All weather values are within valid ranges. ✓")


def check_numeric_stats(df: pd.DataFrame) -> None:
    separator("10. Numeric Column Summary Statistics")
    numeric_cols = [
        "population_density_sqkm", "water_logging_risk_index",
        "sanitation_score", "avg_temp_c", "max_temp_c", "min_temp_c",
        "rainfall_mm", "relative_humidity_pct", "dengue_cases",
    ]
    print(df[numeric_cols].describe().round(3).to_string())


def check_date_coverage_per_ward(df: pd.DataFrame) -> None:
    separator("11. Date Coverage per Ward")
    dates_per_ward = df.groupby("ward_id")["date"].nunique()
    uniform = dates_per_ward.nunique() == 1
    print(f"  All wards have same number of dates : {'YES ✓' if uniform else 'NO'}")
    print(f"  Dates per ward                      : {dates_per_ward.unique().tolist()}")


def run_full_audit() -> None:
    """Run every audit check in sequence."""
    print("\n" + "#" * 60)
    print("#   DENGUARD AI — Dataset Audit")
    print("#" * 60)

    df = load_data()

    check_shape(df)
    check_columns(df)
    check_date_range(df)
    check_cities_wards(df)
    check_missing(df)
    check_duplicates(df)
    check_dengue_distribution(df)
    check_ward_vulnerability(df)
    check_weather_sanity(df)
    check_numeric_stats(df)
    check_date_coverage_per_ward(df)

    separator("AUDIT COMPLETE")
    print("  All checks finished successfully.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_full_audit()
