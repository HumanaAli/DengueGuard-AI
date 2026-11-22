# DENGUARD AI — Final Technical Verification Report

**Document Purpose:** Independent verification of all data pipelines, models, hashes, evaluation metrics, and documentation prior to Stage 4 (Dashboard Development).  
**Verification Date:** 2026-11-21  
**Lead Auditor:** Antigravity AI Technical Partner  

---

## 1. Investigation of Previous Command Failure (Task 1)

| Field | Detail |
| :--- | :--- |
| **Command Summary** | `Audit dataset splits and shift calculation` |
| **Execution Context** | PowerShell invocation using inline `python.exe -c "..."` |
| **Exit Code** | Code 1 |
| **Exact Traceback** | `File "<string>", line 8` <br> `print(f'{s:<6}: obs [{sub[" date\].min()} ...'` <br> `SyntaxError: unterminated string literal (detected at line 8)` |
| **Root Cause Analysis** | **Shell Syntax/Escaping Error**: Windows PowerShell stripped escape characters (`\"`) inside an inline Python nested f-string literal, mangling `sub["date"]` into `sub[" date\]`. |
| **Impact on Project** | **Zero Data / Code Impact**: No assertion failed, no dataset calculation was incorrect, and no model code was affected. When saved into a `.py` script (`src/audit_pipeline.py`) and executed without shell quote interference, the script exited cleanly with Code 0. |
| **Blocks Dashboard?** | **NO** |

---

## 2. Comprehensive Verification Matrix (Tasks 2 – 6)

| # | Check Name | Actual Result | Status | Evidence / Verification Method | Failure Explanation (if any) | Blocks Dashboard? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Raw Dataset Integrity** | 31,320 rows, 12 columns, 60 wards $\times$ 522 Mondays, 0 nulls, 0 duplicates. | **PASS** | `raw.shape == (31320, 12)`, `raw.isnull().sum() == 0`, `raw.duplicated().sum() == 0` | None | NO |
| **2** | **Processed Table Integrity** | 30,840 rows, 41 columns, 60 wards $\times$ 514 Mondays, 0 nulls, 0 duplicates. | **PASS** | `proc.shape == (30840, 41)`, `proc.isnull().sum() == 0`, `proc.duplicated().sum() == 0` | None | NO |
| **3** | **Target 2W Shift Accuracy** | 0 mismatches across all 30,840 rows compared against raw Excel case counts. | **PASS** | `(proc_sorted['target_2w'] == raw_shifted_2w).all()` | None | NO |
| **4** | **Target 4W Shift Accuracy** | 0 mismatches across all 30,840 rows compared against raw Excel case counts. | **PASS** | `(proc_sorted['target_4w'] == raw_shifted_4w).all()` | None | NO |
| **5** | **Target Date Formulations** | 0 date calculation errors for both 2w and 4w target dates. | **PASS** | `target_date_2w == date + 2 weeks`, `target_date_4w == date + 4 weeks` | None | NO |
| **6** | **Chronological Split Separation** | Train ($\le 2023$, 24,540 rows), Val ($2024$, 3,180 rows), Test ($2025$, 3,120 rows). Zero overlap. | **PASS** | Strict partitioning on `target_date_4w`. All train outcomes completed $\le$ 2023-12-25. | None | NO |
| **7** | **Feature Leakage Safeguard** | Exactly 34 features. `dengue_cases`, `target_2w`, `target_4w`, `target_date_*`, `date`, `split` are 100% excluded. | **PASS** | Verified via `m2.feature_names_in_` and `m4.feature_names_in_`. Forbidden list check = `[]`. | None | NO |
| **8** | **2W Model SHA-256 Hash** | `3646673bf04b6002739edfebf619beacfb7bd2b3678026640d1e991df112f8c7` | **PASS** | Matches original model and backup byte-for-byte. | None | NO |
| **9** | **4W Model SHA-256 Hash** | `fb889d62470594ae70602ad9b7b609079d6ee1acad3ba3eb52e5498cdcd3c4fb` | **PASS** | Matches original model and backup byte-for-byte. | None | NO |
| **10** | **Model Preprocessing Fitting** | `OrdinalEncoder` handles unseen categories (`unknown_value=-1`). Passthrough on 32 numericals. | **PASS** | Robustness test passing unknown ward `UNKNOWN_WARD_99` outputs safe baseline ($0.01$ cases). | None | NO |
| **11** | **Threshold Config Integrity** | `models/calibrated_config.json` correctly loaded by `PredictionEngine`. | **PASS** | Thresholds verified: 2w (0.8, 4.0, 8.0), 4w (0.8, 4.0, 8.5). | None | NO |
| **12** | **2W Test Regression Metrics** | $\text{MAE} = 1.0549$, $\text{RMSE} = 2.1256$, $\text{Poisson Deviance} = 0.8576$. | **PASS** | Independently reproduced on 2025 Test Split ($N = 3,120$). | None | NO |
| **13** | **4W Test Regression Metrics** | $\text{MAE} = 1.1904$, $\text{RMSE} = 2.3158$, $\text{Poisson Deviance} = 1.0087$. | **PASS** | Independently reproduced on 2025 Test Split ($N = 3,120$). | None | NO |
| **14** | **2W Test Ordinal Metrics** | Critical Recall: **$72.22\%$**, Critical Precision: **$47.27\%$**, Macro F1: **$0.6183$**. | **PASS** | Independently reproduced on 2025 Test Split. Catches 182 / 252 true outbreaks. | None | NO |
| **15** | **4W Test Ordinal Metrics** | Critical Recall: **$65.08\%$**, Critical Precision: **$38.68\%$**, Macro F1: **$0.5856$**. | **PASS** | Independently reproduced on 2025 Test Split. Catches 164 / 252 true outbreaks. | None | NO |
| **16** | **False Critical Alarm Analysis** | $>86.9\%$ of false alarms come from High ($4-9.9$ cases). Baseline zero-case wards have $<0.25\%$ false alarms. | **PASS** | 2w: 177/203 from High, 4 from Low. 4w: 226/260 from High, 1 from Low. | None | NO |
| **17** | **Real Data Diversity Check** | Real test records trigger all 4 risk tiers (Low, Medium, High, Critical). All predictions non-negative. | **PASS** | Tested across 5 wards and historical dates in `src/prediction_tests.py`. | None | NO |
| **18** | **Documentation Accuracy** | All numbers in `docs/MODEL_EVALUATION.md` and `docs/QUALITY_AUDIT.md` match execution code. | **PASS** | Full audit cross-referencing completed. | None | NO |
| **19** | **Operational Weather Feed Note** | Weekly weather rows represent completed 7-day aggregates. | **WARNING** (Operational Note) | Public health officers must run forecasts at week close (Sunday night / Monday morning) using the completed 7-day meteorological aggregate. | Not a code defect. Must be clearly labeled on dashboard UI. | NO |

---

## 3. Summary of Verified Files

* `data/hackathon_preli_dataset.xlsx`: Verified pristine ($31,320$ rows, untouched).
* `data/processed/model_dataset.csv`: Verified exact chronological shifts and zero leakage ($30,840$ rows).
* `models/model_2w.joblib`: SHA-256 hash verified (`3646673bf0...`).
* `models/backups/model_2w_backup.joblib`: Byte-identical replica verified.
* `models/model_4w.joblib`: SHA-256 hash verified (`fb889d6247...`).
* `models/backups/model_4w_backup.joblib`: Byte-identical replica verified.
* `models/experiments/`: Option B experimental models preserved intact.
* `models/calibrated_config.json`: Adopted operational thresholds verified.
* `src/model_loader.py`: Inference engine verified with zero-leakage column filtering and defensive error handling.
* `src/independent_verification.py`: Standalone, reproducible verification script exiting with Code 0.

---

## 4. Final Recommendation

### **APPROVED FOR STAGE 4 (STREAMLIT DASHBOARD DEVELOPMENT)**

**Justification:**
1. All pipelines and models are 100% reproducible, byte-verified, and free of data leakage.
2. The prior Code 1 error was definitively traced to an inline PowerShell string-escaping artifact, not a code or data defect.
3. Both prediction horizons have confirmed test set performance (MAE: $1.05$ / $1.19$; Critical Recall: $72.2\%$ / $65.1\%$).
4. The loader (`PredictionEngine`) is verified and ready to be integrated into `dashboard/app.py`.
