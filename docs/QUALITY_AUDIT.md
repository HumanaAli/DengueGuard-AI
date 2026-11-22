# DENGUARD AI — Technical Quality Audit Report

**Audit Target:** Complete DenguAlert Codebase & Model Pipeline  
**Auditor:** Antigravity AI Technical Partner  
**Audit Date:** 2026-11-21  
**Project State:** Pre-Dashboard Quality Gate  

---

## Executive Summary & Readiness Verdict

| Audit Domain | Status | Key Evidence |
| :--- | :--- | :--- |
| **1. Dataset & Pipeline Integrity** | **PASS** | 31,320 rows, 0 nulls, perfectly balanced 60 wards $\times$ 522 weeks. Target shifts verified against raw Excel. |
| **2. Feature & Leakage Safety** | **PASS** | All 34 features verified. Current `dengue_cases` and all targets strictly excluded from input vectors. |
| **3. Model & Backup Integrity** | **PASS** | Original models and backup files are **100% byte-identical** (SHA-256 verified). Pipelines load and predict in $<0.2$s. |
| **4. Evaluation & Metric Integrity** | **PASS** | Validation threshold tuning completely isolated from 2025 Test Set. Confusion matrices and metrics independently recomputed and verified. |
| **5. Robustness & Prediction Tests** | **PASS** | Tested across multiple wards, seasons, and edge cases (unseen categories, missing features). All predictions $\ge 0.0$. |
| **6. Documentation & Reproducibility** | **PASS** | Comprehensive technical and quality documentation generated. |

### Final Decision:
**READY FOR DASHBOARD DEVELOPMENT (STAGE 4) — CONDITIONAL UPON REVIEWING DOCUMENTED OPERATIONAL NOTES.**

---

## 1. Detailed Section-by-Section Audit

### Section 1: Dataset and Data Pipeline Audit
* **Status:** **PASS**
* **Files Inspected:**
  * `data/hackathon_preli_dataset.xlsx`
  * `src/feature_engineering.py`
  * `data/processed/model_dataset.csv`
* **Findings:**
  1. **Dataset Balance:** Raw Excel contains exactly 31,320 rows (60 wards $\times$ 522 weeks, 2016-01-04 to 2025-12-29).
  2. **Target Calculation:** `target_2w` and `target_4w` are generated using grouped negative shifts (`df.groupby('ward_id')['dengue_cases'].shift(-2)` and `.shift(-4)`). Spot checks across multiple wards confirmed exact parity with raw future case counts.
  3. **Target Dates:** Formulated strictly as `date + pd.Timedelta(weeks=2)` and `date + pd.Timedelta(weeks=4)`.
  4. **Chronological Split Integrity:**
     * Train: `target_date_4w` $\le$ `2023-12-25` ($N = 24,540$).
     * Validation: `target_date_4w` in $2024$ ($N = 3,180$).
     * Test: `target_date_4w` in $2025$ ($N = 3,120$).
     * Zero temporal overlap across split boundaries.
  5. **Current-Week Weather Availability (Operational Nuance):**
     * The dataset rows are timestamped on Monday (e.g. `2016-01-04`).
     * `rainfall_mm`, `avg_temp_c`, etc., represent weekly summaries.
     * *Operational Note:* In an active real-time municipal deployment, these features represent the completed 7-day weather summary preceding the Monday forecast date. When deploying the dashboard, municipal officers must run forecasts at the close of the week (Sunday night / Monday morning) using the completed week's meteorological feed.

---

### Section 2: Feature Audit (34 Model Features)
* **Status:** **PASS**
* **Verification:**
  * All 34 features documented in [`docs/PROJECT_TECHNICAL_DOCUMENTATION.md`](file:///d:/Dengu_Alert/docs/PROJECT_TECHNICAL_DOCUMENTATION.md).
  * Excluded forbidden columns verified: `dengue_cases`, `target_2w`, `target_4w`, `target_date_2w`, `target_date_4w`, `date`, `split`.
  * Multi-week rainfall lags (`rain_lag1w` through `rain_lag4w`) and shifted rolling sums (`rain_roll2w`, `rain_roll4w`) are calculated strictly within wards. No forward-looking observations are used.

---

### Section 3: Model and Backup Integrity
* **Status:** **PASS**
* **SHA-256 Checksums (Byte-Identical Verification):**
  * `models/model_2w.joblib`:
    `3646673bf04b6002739edfebf619beacfb7bd2b3678026640d1e991df112f8c7`
  * `models/backups/model_2w_backup.joblib`:
    `3646673bf04b6002739edfebf619beacfb7bd2b3678026640d1e991df112f8c7`
    *(MATCH: 100% BYTE-IDENTICAL)*
  * `models/model_4w.joblib`:
    `fb889d62470594ae70602ad9b7b609079d6ee1acad3ba3eb52e5498cdcd3c4fb`
  * `models/backups/model_4w_backup.joblib`:
    `fb889d62470594ae70602ad9b7b609079d6ee1acad3ba3eb52e5498cdcd3c4fb`
    *(MATCH: 100% BYTE-IDENTICAL)*
* **Loader Functionality:**
  * `PredictionEngine` in `src/model_loader.py` dynamically reads thresholds from `models/calibrated_config.json`.
  * ColumnTransformer binds features by name and raises `ValueError` if any required feature is missing.

---

### Section 4: Evaluation & Metric Integrity
* **Status:** **PASS**
* **Independent Verification of Claims in `docs/MODEL_EVALUATION.md`:**
  1. *Claim:* 2025 Test Set was not used for threshold tuning.  
     *Verified:* Threshold sweeps were executed strictly on the 2024 validation dataset ($N = 3,180$).
  2. *Claim:* 2-Week Critical Recall on Test Set is $72.2\%$, capturing $182/252$ true outbreaks.  
     *Verified:* Independently confirmed on 2025 test split.
  3. *Claim:* 4-Week Critical Recall on Test Set is $65.1\%$, capturing $164/252$ true outbreaks.  
     *Verified:* Independently confirmed on 2025 test split.
  4. *Claim:* Regression MAE and RMSE are not altered by threshold calibration.  
     *Verified:* Continuous case predictions remain strictly at $\text{MAE} = 1.0549$ (2w) and $1.1904$ (4w). Threshold calibration only alters the categorical decision mapping.
  5. *Claim:* False Critical alarms occur predominantly in High transmission zones.  
     *Verified:* $87.2\%$ (2w) and $86.9\%$ (4w) of all false Critical alarms occur in wards with actual cases between $4$ and $9.9$ cases. Zero-case baseline wards have $<0.2\%$ false alarm rates.

---

### Section 5: Prediction Tests
* **Status:** **PASS**
* **Script:** `src/prediction_tests.py`
* **Findings:**
  * Tested across diverse wards (`CCC_Ward_01`, `DSCC_Ward_01`, `DNCC_Ward_03`, `GCC_Ward_01`, `NCC_Ward_02`).
  * Real observations triggered all 4 tiers: Low, Medium, High, and Critical.
  * Verified $100\%$ non-negative predictions across all test samples.
  * Robustness against unseen categories verified: Passing an unknown ward or city routes to `unknown_value=-1` in `OrdinalEncoder`, outputting a safe baseline prediction without crashing.
  * Dropping required columns correctly triggers an explicit `ValueError`.

---

## 2. Inventory of Inspected Files

1. `data/hackathon_preli_dataset.xlsx` (Original Dataset)
2. `data/processed/model_dataset.csv` (Processed ML Table)
3. `src/data_audit.py` (Data Audit Pipeline)
4. `src/feature_engineering.py` (Feature Engineering Pipeline)
5. `src/train.py` (Model Training Pipeline)
6. `src/model_loader.py` (Operational Loader)
7. `src/validate_comparison.py` (Validation Benchmarking Script)
8. `src/evaluate_test_candidate.py` (One-Time Test Evaluation Script)
9. `src/prediction_tests.py` (Live Prediction Test Script)
10. `models/model_2w.joblib` (Production 2w Model)
11. `models/model_4w.joblib` (Production 4w Model)
12. `models/backups/model_2w_backup.joblib` (Verified Backup)
13. `models/backups/model_4w_backup.joblib` (Verified Backup)
14. `models/experiments/model_2w_weighted.joblib` (Preserved Option B)
15. `models/experiments/model_4w_weighted.joblib` (Preserved Option B)
16. `models/calibrated_config.json` (Adopted Threshold Configuration)
17. `docs/DATA_AUDIT.md` (Stage 1 Audit Report)
18. `docs/FEATURE_ENGINEERING.md` (Stage 2 Feature Engineering Report)
19. `docs/MODEL_EVALUATION.md` (Stage 3 Evaluation Report)
20. `docs/MODEL_IMPROVEMENT.md` (Stage 3.1 Optimization Report)

---

## 3. Discovered Code / Operational Nuances (For User Awareness)

1. **Explicit Feature Ordering in `src/model_loader.py`:**  
   *Current behavior:* `ColumnTransformer` matches columns by name, so reordered DataFrames work correctly.  
   *Recommendation for Dashboard:* To make the loader even more defensive against user-uploaded CSVs, `PredictionEngine.predict()` should enforce the exact 34 column ordering explicitly before calling `model.predict()`. (Proposed for inclusion in Stage 4 dashboard loader).
2. **Current-Week Meteorological Availability:**  
   *Operational Reality:* Real-time public health predictions require completed 7-day weather aggregates. In the dashboard, we should clearly display an "Observation Period: Week Ending [Date]" indicator so users know the weather inputs reflect the completed past 7-day window.

---

## 4. Final Verdict

The data pipeline, feature engineering, trained models, backup integrity, calibrated configuration, and evaluation statistics are **empirically verified, reproducible, and robust**.

**The project is certified ready for Stage 4: Streamlit Dashboard Development.**
