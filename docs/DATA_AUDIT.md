# DENGUARD AI — Data Audit Report

**Project:** DENGUARD AI — Urban Dengue Early-Warning System  
**Hackathon:** Preliminary Round (Nov 21, 2026)  
**Audit script:** `src/data_audit.py`  
**Dataset file:** `data/hackathon_preli_dataset.xlsx`  
**Audit date:** 2026-11-21

---

## 1. Files Inspected

| File | Size | Notes |
|------|------|-------|
| `data/hackathon_preli_dataset.xlsx` | ~2.0 MB | Single Excel file, one sheet |
| `doc/Hackathon_preli_nov21.pdf` | 342 KB | Hackathon brief (custom-encoded fonts; text not machine-extractable) |
| `inspect_dataset.py` | 779 B | Existing starter script (project root, uses hard-coded path) |
| `src/data_audit.py` | — | **New:** comprehensive, reusable audit script |

> [!NOTE]
> The PDF uses embedded custom fonts (CIDFontType2 with Unicode mapping). Its title is **"Preliminary Round Brief"** and it links to a Google Sheets resource. The PDF cannot be read programmatically without a PDF library (none installed). Key requirements below are inferred from the project description and dataset structure.

---

## 2. Dataset Overview

| Property | Value |
|----------|-------|
| Sheet name | `bangladesh_dengue_large_dataset` |
| Rows | **31,320** |
| Columns | **12** |
| Date range | **2016-01-04 → 2025-12-29** (10 years) |
| Temporal resolution | **Weekly** (every 7 days, perfectly regular) |
| Cities | **5** |
| Wards | **60** total |
| Structure check | 60 wards × 522 weeks = **31,320 rows ✓** |

---

## 3. Column Names and Data Types

| Column | Type | Missing | Unique Values | Description |
|--------|------|---------|---------------|-------------|
| `date` | datetime64 | 0 | 522 | Weekly timestamp |
| `city` | str | 0 | 5 | City name |
| `ward_id` | str | 0 | 60 | Unique ward identifier |
| `population_density_sqkm` | int64 | 0 | 60 | People per km² |
| `water_logging_risk_index` | float64 | 0 | 41 | 0–1 flood-risk score |
| `sanitation_score` | float64 | 0 | 41 | 0–1 sanitation quality |
| `avg_temp_c` | float64 | 0 | 1,851 | Weekly average temperature (°C) |
| `max_temp_c` | float64 | 0 | 437 | Weekly max temperature (°C) |
| `min_temp_c` | float64 | 0 | 453 | Weekly min temperature (°C) |
| `rainfall_mm` | float64 | 0 | 10,079 | Weekly rainfall (mm) |
| `relative_humidity_pct` | float64 | 0 | 4,996 | Relative humidity (%) |
| `dengue_cases` | int64 | 0 | 38 | Weekly reported cases (**target variable**) |

---

## 4. Date Range and Temporal Structure

- **Start:** 2016-01-04 (Monday)
- **End:** 2025-12-29
- **Total weeks:** 522 (all exactly 7 days apart — perfectly regular, no gaps)
- **Coverage:** 10 full years of weekly data for all 60 wards simultaneously

> [!IMPORTANT]
> The dataset is a **balanced panel**: every ward has exactly 522 observations covering the full range. There are no missing time periods for any ward.

---

## 5. Cities and Wards

| City | Wards | Sample Ward IDs |
|------|-------|-----------------|
| Dhaka North | 20 | DNCC_Ward_01 … DNCC_Ward_20 |
| Dhaka South | 20 | DSCC_Ward_01 … DSCC_Ward_20 |
| Chattogram | 10 | CCC_Ward_01 … CCC_Ward_10 |
| Gazipur | 5 | GCC_Ward_01 … GCC_Ward_05 |
| Narayanganj | 5 | NCC_Ward_01 … NCC_Ward_05 |

---

## 6. Missing Values

**None.** All 12 columns across all 31,320 rows are fully populated. ✓

---

## 7. Duplicate Records

| Check | Result |
|-------|--------|
| Exact duplicate rows | **0** ✓ |
| Duplicate city + ward + date combinations | **0** ✓ |

---

## 8. Dengue Cases Distribution (Target Variable)

| Statistic | Value |
|-----------|-------|
| Total cases (sum) | **82,572** |
| Zero-case rows | **18,176** (58.0%) |
| Non-zero rows | **13,144** (42.0%) |
| Minimum | 0 |
| Maximum | **41** |
| Mean | 2.636 |
| Median | **0.0** |
| Std deviation | 4.608 |
| Distinct count values | 38 (0 through 41) |

> [!WARNING]
> **Class imbalance / zero-inflation:** 58% of records have zero cases. The distribution is heavily right-skewed (median = 0, mean = 2.6, max = 41). Standard regression models may predict zero poorly. We will need strategies like:
> - Zero-inflated models or Tweedie regression
> - Poisson / Negative-Binomial regression (counts data)
> - Threshold-based outbreak classification (binary or multi-class)

**Cases by City:**

| City | Total | Mean/week | Max | Std |
|------|-------|-----------|-----|-----|
| Dhaka South | 27,193 | 2.60 | 41 | 4.52 |
| Dhaka North | 27,001 | 2.59 | 39 | 4.56 |
| Chattogram | 13,470 | 2.58 | 38 | 4.62 |
| Gazipur | 8,295 | **3.18** | 34 | 5.30 |
| Narayanganj | 6,613 | 2.53 | 27 | 4.34 |

Gazipur has the highest mean case rate despite having only 5 wards.

---

## 9. Ward Vulnerability Features

All three vulnerability features are **static per ward** (they do not change over time):

| Feature | Min | Max | Description |
|---------|-----|-----|-------------|
| `population_density_sqkm` | 18,314 | 74,252 | Varies widely across wards |
| `water_logging_risk_index` | 0.22 | 0.94 | 0 = no risk, 1 = highest risk |
| `sanitation_score` | 0.25 | 0.88 | 0 = poor, 1 = excellent |

> [!NOTE]
> Since these features don't change over time, they act as **ward-level metadata** (fixed effects). They are valuable for distinguishing high-risk from low-risk wards but won't help predict _when_ outbreaks happen within a ward — only the weather features carry temporal signal.

---

## 10. Numeric Feature Ranges

| Feature | Min | 25th% | Median | 75th% | Max |
|---------|-----|-------|--------|-------|-----|
| avg_temp_c | 15.74 | 22.88 | 26.85 | 31.02 | 37.51 |
| max_temp_c | 23.12 | 27.85 | 31.51 | 35.99 | 40.79 |
| min_temp_c | 13.06 | 18.00 | 21.67 | 25.97 | 31.64 |
| rainfall_mm | 0.00 | 0.92 | 8.37 | 59.42 | **268.83** |
| relative_humidity_pct | 41.84 | 60.47 | 68.22 | 83.98 | 99.00 |
| dengue_cases | 0 | 0 | 0 | 4 | 41 |

Rainfall is highly skewed (median 8 mm vs max 269 mm) — log transformation will likely be useful.

---

## 11. Data Quality Checks

| Check | Result |
|-------|--------|
| avg_temp between min and max | ✓ All 31,320 rows valid |
| Negative rainfall | ✓ None |
| Humidity outside 0–100% | ✓ None |
| Negative dengue cases | ✓ None |

**No data quality issues detected.**

---

## 12. Problems Discovered

| # | Issue | Severity | Implication |
|---|-------|----------|-------------|
| 1 | **58% zero cases** (zero-inflated target) | Medium | Need count-data models or classification threshold |
| 2 | **Rainfall highly right-skewed** (max 269 mm) | Low | Log-transform or clip before modeling |
| 3 | **Ward vulnerability is time-invariant** | Low | Good for ward profiling, not for temporal prediction |
| 4 | **PDF not machine-readable** (custom font encoding) | Low | Could not auto-extract hackathon criteria; requirements inferred |
| 5 | **Existing `inspect_dataset.py` uses hard-coded path** | Low | Will break if run from a different folder |

---

## 13. What We Should Build Next

```
Phase 1 → Feature Engineering
  - Lag features: dengue cases at week t-2, t-3, t-4 (the 2–4 week prediction window)
  - Rolling weather averages: 2-week, 4-week rolling mean for temp, rainfall, humidity
  - Seasonality: week_of_year, month, monsoon_flag
  - Ward × weather interaction terms

Phase 2 → Model Training
  - Baseline: Poisson regression (interpretable, handles count data)
  - Main model: LightGBM or XGBoost (handles zero-inflation well)
  - Optional: LSTM/Transformer for sequence modeling
  - Predict dengue_cases 2 and 4 weeks ahead

Phase 3 → Risk Scoring
  - Convert predicted counts to outbreak risk (Low/Medium/High)
  - Rank all 60 wards by predicted risk each week

Phase 4 → Dashboard (Streamlit)
  - Ward-level map showing risk level
  - 2–4 week forecast chart
  - Feature importance / model explanation
```

> [!TIP]
> The dataset is well-structured, clean, and balanced — an excellent foundation. The main challenge is the **zero-inflated, count-valued target** and designing good **lag/rolling features** to capture the 2–4 week lag between weather events and dengue outbreaks.

---

*Report auto-generated from `src/data_audit.py` output — all numbers verified against actual script run on 2026-11-21.*
