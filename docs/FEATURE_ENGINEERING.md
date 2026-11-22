# DENGUARD AI — Feature Engineering Report

**Script:** `src/feature_engineering.py`  
**Input:** `data/hackathon_preli_dataset.xlsx`  
**Output:** `data/processed/model_dataset.csv`  
**Run date:** 2026-11-21 — Exit code 0 ✓

---

## 1. What Feature Engineering Means (Beginner Explanation)

The raw dataset tells us: "On week X, Ward Y had these weather conditions and this many dengue cases."

But a prediction model needs to see the **past** in order to predict the **future**. Feature engineering adds new columns that look backward and forward in time:

- **Lag features** = "What was the rainfall 2 weeks ago?" (looking backward — safe to use as model input)
- **Rolling features** = "What was the total rainfall over the past 4 weeks?" (also backward — safe)
- **Target columns** = "How many dengue cases will there be 2 weeks from now?" (looking forward — only used as the label to train against, never as an input)

---

## 2. Why One Combined Dataset (Not Two Separate Files)

We created **one file** (`model_dataset.csv`) with both `target_2w` and `target_4w` as separate columns. Reasons:

- All features are identical for both prediction horizons — no need to duplicate.
- During model training, we simply choose which target column to use (`target_2w` for the 2-week model, `target_4w` for the 4-week model).
- Keeps the project tidy: one processed file to track, version and load.

---

## 3. Data Leakage Prevention — The Most Important Rule

> **Data leakage** means accidentally giving the model information it wouldn't have in real life. If we did this, the model would score perfectly in testing but fail completely when deployed.

Here is how we prevent every type of leakage:

| Risk | Prevention |
|------|-----------|
| Using today's weather to predict today | All weather features are raw (available today) or lagged (past only) |
| Using future weather as a feature | Never done — only `rainfall_mm` at the current observation date, plus lags ≥ 1 week |
| Using today's cases to predict today | Minimum dengue lag is `dengue_lag1w` (previous week) |
| Cross-ward contamination | All shifts and rolling windows are computed inside `groupby("ward_id")` |
| Future target bleeding into train | Split is done on **target date** (not observation date) |
| Partial rolling sums | `min_periods=N` ensures rolling windows only compute when fully filled |

---

## 4. Feature Groups Created

### 4.1 Rainfall Features (7 columns)

| Column | Meaning | How Created |
|--------|---------|-------------|
| `rain_lag1w` | Rainfall 1 week ago | `shift(1)` within ward |
| `rain_lag2w` | Rainfall 2 weeks ago | `shift(2)` within ward |
| `rain_lag3w` | Rainfall 3 weeks ago | `shift(3)` within ward |
| `rain_lag4w` | Rainfall 4 weeks ago | `shift(4)` within ward |
| `rain_roll2w` | Total rainfall over past 2 weeks | `shift(1)` then `rolling(2).sum()` |
| `rain_roll4w` | Total rainfall over past 4 weeks | `shift(1)` then `rolling(4).sum()` |
| `rain_log1p` | Log(1 + current rainfall) | `np.log1p()` to reduce skew |

> **Why rolling sums?** A single rainy week might not cause an outbreak, but 4 consecutive rainy weeks creates sustained breeding conditions. Rolling sums capture this cumulative effect.
>
> **Why log transform?** Rainfall ranged from 0 to 269 mm (median = 8 mm). The huge outliers would dominate the model. `log1p` compresses the scale while keeping the ordering.

### 4.2 Temperature and Humidity Features (8 columns)

| Column | Meaning | How Created |
|--------|---------|-------------|
| `temp_lag1w` | Average temperature 1 week ago | `shift(1)` within ward |
| `temp_lag2w` | Average temperature 2 weeks ago | `shift(2)` within ward |
| `temp_lag3w` | Average temperature 3 weeks ago | `shift(3)` within ward |
| `temp_lag4w` | Average temperature 4 weeks ago | `shift(4)` within ward |
| `temp_roll4w_mean` | Mean temperature over past 4 weeks | `shift(1)` then `rolling(4).mean()` |
| `humidity_lag1w` | Humidity 1 week ago | `shift(1)` within ward |
| `humidity_lag2w` | Humidity 2 weeks ago | `shift(2)` within ward |
| `temp_x_humidity_lag2w` | Temperature × Humidity at lag 2 | `temp_lag2w × (humidity_lag2w / 100)` |

> **Why the interaction feature?** Mosquitoes breed optimally in warm AND humid conditions simultaneously. Neither temperature alone nor humidity alone captures this — their product does.

### 4.3 Historical Dengue Lag Features (4 columns)

| Column | Meaning |
|--------|---------|
| `dengue_lag1w` | Cases 1 week ago (strongest predictor) |
| `dengue_lag2w` | Cases 2 weeks ago |
| `dengue_lag3w` | Cases 3 weeks ago |
| `dengue_lag4w` | Cases 4 weeks ago |

> These are typically the **most predictive features** in dengue models. Outbreaks have momentum — high case counts last week strongly predict high counts this week.

### 4.4 Seasonality Features (5 columns)

| Column | Meaning |
|--------|---------|
| `week_of_year` | 1–52, captures annual cycle |
| `month` | 1–12 |
| `year` | 2016–2025 |
| `is_monsoon` | 1 if June–September, else 0 |
| `is_post_monsoon` | 1 if October–November, else 0 |

### 4.5 Original Columns (12, unchanged)

`date`, `city`, `ward_id`, `population_density_sqkm`, `water_logging_risk_index`, `sanitation_score`, `avg_temp_c`, `max_temp_c`, `min_temp_c`, `rainfall_mm`, `relative_humidity_pct`, `dengue_cases`

---

## 5. Future Prediction Targets

### How negative-shift creates future targets

```
Ward CCC_Ward_01 dengue_cases (original):
  Week 1 (2016-01-04): 0 cases
  Week 2 (2016-01-11): 0 cases
  Week 3 (2016-01-18): 0 cases
  Week 4 (2016-01-25): 2 cases
  Week 5 (2016-02-01): 3 cases  ← target_2w for Week 3
  Week 7 (2016-02-15): 5 cases  ← target_4w for Week 3

After shift(-2):   target_2w for Week 3 = 3  ✓
After shift(-4):   target_4w for Week 3 = 5  ✓
```

The model is given data at Week 3 (observation date) and must predict what will happen at Week 5 (target_2w) and Week 7 (target_4w).

| Column | Description |
|--------|-------------|
| `target_2w` | Dengue cases at observation date + 2 weeks |
| `target_4w` | Dengue cases at observation date + 4 weeks |
| `target_date_2w` | Calendar date of target_2w |
| `target_date_4w` | Calendar date of target_4w (used for split) |

---

## 6. Row Counts After Dropping Incomplete Rows

Each ward loses exactly 8 rows:
- **4 rows at the start**: lags 1–4 produce NaN (no data before the dataset begins)
- **4 rows at the end**: `target_4w` requires 4 future rows that don't exist

```
Before dropping: 31,320 rows (60 wards × 522 weeks)
After  dropping: 30,840 rows (60 wards × 514 weeks)
Dropped:            480 rows (8 per ward × 60 wards)
```

---

## 7. Chronological Train / Validation / Test Split

### Why chronological (not random)?

If we split randomly, a training row from 2024 could appear next to a test row from 2023. The model would essentially see the future during training — a form of leakage. Time-series data must always be split in time order.

### Split key: `target_date_4w`

We split on the **4-week target date** (the later of the two targets). This guarantees that every training row has both its `target_2w` AND `target_4w` outcomes falling before 2024.

```
                              target_date_4w
                 ┌──────────────────────────────────────────────┐
TRAIN    ────────│─────────────────────────── ≤ 2023-12-31      │
VAL              │   2024-01-01 ──────────── 2024-12-30         │
TEST             │                 2025-01-06 ──────── 2025-12-29 │
                 └──────────────────────────────────────────────┘
```

### Split results

| Split | Observation dates | Target dates (4w) | Rows |
|-------|-------------------|-------------------|------|
| **train** | 2016-02-01 → 2023-11-27 | 2016-02-29 → 2023-12-25 | 24,540 |
| **val** | 2023-12-04 → 2024-12-02 | 2024-01-01 → 2024-12-30 | 3,180 |
| **test** | 2024-12-09 → 2025-12-01 | 2025-01-06 → 2025-12-29 | 3,120 |
| **Total** | | | **30,840** |

---

## 8. Verification Results (all passed ✓)

| Check | Result |
|-------|--------|
| **V1** No NaN in features or targets | ✓ 0 NaN values found |
| **V2** Lag alignment (CCC_Ward_01, row 10) | ✓ rain_lag1w = 12.57, rain_lag2w = 0.00 — both match manual lookup |
| **V3** Target alignment (CCC_Ward_01, row 10) | ✓ target_2w and target_4w match cases at t+2 and t+4 |
| **V4** No cross-ward contamination | ✓ Ward 01 and Ward 02 boundary values differ as expected |
| **V5** Split boundaries correct | ✓ Train ends ≤ 2023-12-25, val in 2024, test in 2025 |
| **V6** Feature count correct | ✓ 36 input features, 2 targets, 3 metadata columns |
| **V7** Row count correct | ✓ 30,840 = 60 × 514 |

---

## 9. Final Dataset Summary

| Property | Value |
|----------|-------|
| File | `data/processed/model_dataset.csv` |
| Size | ~8.3 MB |
| Rows | 30,840 |
| Columns | 41 |
| Input features | 36 |
| Target columns | `target_2w`, `target_4w` |
| Split column | `split` (train / val / test) |
| NaN values | **0** |

---

## 10. What We Should Build Next

```
Step 1 (complete ✓): Dataset Audit
Step 2 (complete ✓): Feature Engineering → model_dataset.csv

Step 3: Model Training
  ├── Load model_dataset.csv
  ├── Use split column for train/val/test separation
  ├── Train Model A: predict target_2w (2-week horizon)
  ├── Train Model B: predict target_4w (4-week horizon)
  ├── Algorithm: LightGBM (handles zero-inflated counts well)
  ├── Metric: MAE + RMSE (count prediction) + AUROC (outbreak classification)
  └── Feature importance analysis

Step 4: Risk Scoring
  └── Convert predicted case counts → Low / Medium / High risk per ward

Step 5: Dashboard (Streamlit)
  └── Show forecast + risk map for all 60 wards
```

> [!IMPORTANT]
> The `dengue_cases` column in the output is the **current week's observed cases** (kept for reference). It is **not** a valid input feature for prediction — only the lag versions (`dengue_lag1w` through `dengue_lag4w`) should be used as model inputs.

---

*All numbers verified against actual script output — 2026-11-21*
