# DENGUARD AI — Model Training & Evaluation Report

**Stage:** Stage 3 & 3.1 — Outbreak Prediction Engine  
**Algorithm:** HistGradientBoostingRegressor (scikit-learn) with Poisson Loss  
**Input Features:** 34 dynamic & static features (strictly excluding current `dengue_cases` and future targets)  
**Prediction Horizons:**  
1. **2 Weeks Ahead (`target_2w`)**  
2. **4 Weeks Ahead (`target_4w`)**  
**Adopted Strategy:** Option A — Threshold-Calibrated Operational Configuration  

---

## 1. System Overview & Leakage Prevention Safeguards

To strictly adhere to zero-leakage epidemiological modeling standards:
* **Chronological Split Preserved:**
  * **Train Set (2016–2023):** 24,540 observations.
  * **Validation Set (2024):** 3,180 observations (used for hyperparameter selection and threshold calibration).
  * **Test Set (2025):** 3,120 observations (strictly untouched until final testing).
* **Forbidden Columns Excluded from $\mathbf{X}$:**
  * `dengue_cases` (current week observed cases) was completely excluded. The model only receives historical lag signals (`dengue_lag1w` through `dengue_lag4w`).
  * `target_2w`, `target_4w`, `target_date_2w`, `target_date_4w`, `date`, `split` are excluded from input vectors.
* **Preprocessing Isolation:**
  * `OrdinalEncoder` for `city` and `ward_id` was fitted strictly on training data inside a `Pipeline` to prevent category leakage.

---

## 2. Decision Thresholds & Risk Definitions

### Actual Ground-Truth Risk Definitions (Preserved):
* **Low:** Actual cases $< 1.0$ ($y = 0$)
* **Medium:** Actual cases $1.0 \le y < 4.0$ ($y \in \{1, 2, 3\}$)
* **High:** Actual cases $4.0 \le y < 10.0$ ($y \in \{4, \dots, 9\}$)
* **Critical:** Actual cases $\ge 10.0$ ($y \ge 10$)

### Calibrated Operational Prediction Thresholds (Option A):
Due to regression shrinkage caused by the $57\%$ zero-inflation in historical training data, extreme outbreak peaks are compressed slightly toward the mean. Decision thresholds were calibrated on the 2024 validation data to eliminate peak-outbreak misses:
* **2-Week Horizon:** Low: $< 0.8$, Medium: $0.8$ to $< 4.0$, High: $4.0$ to $< 8.0$, Critical: $\ge 8.0$
* **4-Week Horizon:** Low: $< 0.8$, Medium: $0.8$ to $< 4.0$, High: $4.0$ to $< 8.5$, Critical: $\ge 8.5$

---

## 3. Final Benchmark Test Results (Untouched 2025 Test Set, $N = 3,120$)

### A. Continuous Count Regression Metrics
| Metric | 2-Week Ahead (`target_2w`) | 4-Week Ahead (`target_4w`) | Public Health Interpretation |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **1.0549 cases** | **1.1904 cases** | Average prediction error per ward-week |
| **Root Mean Squared Error (RMSE)** | **2.1256 cases** | **2.3158 cases** | Penalizes severe peak estimation spikes |
| **Mean Poisson Deviance** | **0.8576** | **1.0087** | Goodness-of-fit metric for count data |

---

### B. Outbreak Risk Classification Metrics (Before vs. After Calibration)

| Prediction Horizon | Metric | Uncalibrated Baseline (Stage 3) | Option A Calibrated (Stage 3.1) | Improvement ($\Delta$) |
| :--- | :--- | :--- | :--- | :--- |
| **2-Week Ahead** | **Critical Recall** | **47.6%** ($120/252$) | **72.2%** ($182/252$) | **+24.6% (+62 outbreaks caught)** |
| | **Critical Precision** | 53.3% | 47.3% | $-6.0\%$ |
| | **Critical F1-Score** | 0.5034 | **0.5714** | **+0.0680** |
| | **Medium Recall** | 18.6% | **25.3%** | **+6.7%** |
| | **Medium Precision** | 57.8% | 57.1% | Stable |
| | **Low Recall** | 98.3% | **97.1%** | Resilient baseline ($>97\%$) |
| | **Macro F1-Score** | 0.5960 | **0.6183** | **+0.0223** |
| **4-Week Ahead** | **Critical Recall** | **48.4%** ($122/252$) | **65.1%** ($164/252$) | **+16.7% (+42 outbreaks caught)** |
| | **Critical Precision** | 44.7% | 38.7% | $-6.0\%$ |
| | **Critical F1-Score** | 0.4648 | **0.4852** | **+0.0204** |
| | **Medium Recall** | 18.0% | **29.0%** | **+11.0%** |
| | **Medium Precision** | 56.6% | 53.3% | Stable |
| | **Low Recall** | 97.8% | **94.7%** | Resilient baseline ($>94\%$) |
| | **Macro F1-Score** | 0.5763 | **0.5856** | **+0.0093** |

---

### C. Final Confusion Matrices (2025 Test Set)

#### 2-Week Ahead Test Confusion Matrix ($N = 3,120$):
| Actual \ Predicted | Low ($<0.8$) | Medium ($0.8-3.9$) | High ($4.0-7.9$) | Critical ($\ge 8.0$) | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Low** ($<1$) | **1,777** | 47 | 3 | 4 | 1,831 |
| **Medium** ($1-3$) | 219 | **121** | 116 | 22 | 478 |
| **High** ($4-9$) | 0 | 44 | **338** | 177 | 559 |
| **Critical** ($\ge 10$) | 0 | 0 | 70 | **182** | 252 |

* **False Alarm Integrity:** Out of $203$ false Critical predictions, **$177$ ($87.2\%$)** come from the active **High** category ($4$ to $9.9$ cases). Only $4$ out of $1,831$ zero-case wards ($0.2\%$) received a false critical alarm.

#### 4-Week Ahead Test Confusion Matrix ($N = 3,120$):
| Actual \ Predicted | Low ($<0.8$) | Medium ($0.8-3.9$) | High ($4.0-8.4$) | Critical ($\ge 8.5$) | Support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Low** ($<1$) | **1,733** | 88 | 8 | 1 | 1,830 |
| **Medium** ($1-3$) | 193 | **139** | 114 | 33 | 479 |
| **High** ($4-9$) | 2 | 33 | **298** | 226 | 559 |
| **Critical** ($\ge 10$) | 0 | 1 | 87 | **164** | 252 |

* **False Alarm Integrity:** Out of $260$ false Critical predictions, **$226$ ($86.9\%$)** come from **High** ($4$ to $9.9$ cases). **Only $1$ out of $1,830$ zero-case wards ($0.05\%$)** received a false critical alarm.

---

## 4. Production Artifacts & Loading API

1. **Production Pipeline Artifacts:**
   * [`models/model_2w.joblib`](file:///d:/Dengu_Alert/models/model_2w.joblib)
   * [`models/model_4w.joblib`](file:///d:/Dengu_Alert/models/model_4w.joblib)
2. **Backup Artifacts:**
   * `models/backups/model_2w_backup.joblib`
   * `models/backups/model_4w_backup.joblib`
3. **Experimental Artifacts (Preserved):**
   * `models/experiments/model_2w_weighted.joblib`
   * `models/experiments/model_4w_weighted.joblib`
4. **Configuration Specification:**
   * [`models/calibrated_config.json`](file:///d:/Dengu_Alert/models/calibrated_config.json)
5. **Unified Dashboard Loader:**
   * [`src/model_loader.py`](file:///d:/Dengu_Alert/src/model_loader.py): Provides `PredictionEngine` with automatic calibrated risk mapping and UI styling metadata.
