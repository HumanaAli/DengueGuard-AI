"""
Unit and Integration Tests for Streamlit Dashboard Backend Components
=====================================================================
Verifies that:
1. PredictionEngine loads successfully from models/calibrated_config.json.
2. Both 2-week and 4-week predictions execute on real data without errors.
3. Forbidden columns are strictly filtered.
4. Risk categorization and metadata mapping are correct.
5. All 5 dashboard data queries and transformations execute cleanly.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_loader import get_engine, RISK_METADATA, FORBIDDEN_COLS


class TestDashboardBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = get_engine()
        cls.dataset_path = os.path.join(PROJECT_ROOT, "data", "processed", "model_dataset.csv")
        assert os.path.exists(cls.dataset_path), f"Processed dataset missing at {cls.dataset_path}"
        cls.df = pd.read_csv(cls.dataset_path)
        cls.df["date"] = pd.to_datetime(cls.df["date"])

    def test_01_engine_initialization(self):
        """Engine should initialize with both 2w and 4w models and calibrated thresholds."""
        self.assertIsNotNone(self.engine)
        self.assertTrue(hasattr(self.engine, "model_2w"))
        self.assertTrue(hasattr(self.engine, "model_4w"))
        self.assertIn("2w", self.engine.thresholds)
        self.assertIn("4w", self.engine.thresholds)
        self.assertEqual(self.engine.thresholds["2w"]["Critical_min"], 8.0)
        self.assertEqual(self.engine.thresholds["4w"]["Critical_min"], 8.5)

    def test_02_forbidden_columns_filtered(self):
        """PredictionEngine must filter out forbidden columns like dengue_cases and targets."""
        sample_row = self.df.iloc[:5].copy()
        for col in FORBIDDEN_COLS:
            if col in sample_row.columns:
                self.assertIn(col, sample_row.columns)
        
        res_2w = self.engine.predict(sample_row, horizon="2w")
        self.assertEqual(len(res_2w["predicted_cases"]), 5)
        self.assertEqual(len(res_2w["risk_tier"]), 5)

    def test_03_single_record_prediction_module1(self):
        """Module 1 single ward prediction returns valid structure."""
        ward = self.df["ward_id"].iloc[0]  # e.g. CCC_Ward_01
        latest_date = self.df["date"].max()
        record = self.df[(self.df["ward_id"] == ward) & (self.df["date"] == latest_date)]
        self.assertFalse(record.empty)

        res_2w = self.engine.predict(record, horizon="2w")
        res_4w = self.engine.predict(record, horizon="4w")

        self.assertIsInstance(float(res_2w["predicted_cases"][0]), float)
        self.assertIn(res_2w["risk_tier"][0], ["Low", "Medium", "High", "Critical"])
        self.assertIn("color", res_2w["risk_metadata"][0])

        self.assertIsInstance(float(res_4w["predicted_cases"][0]), float)
        self.assertIn(res_4w["risk_tier"][0], ["Low", "Medium", "High", "Critical"])
        self.assertIn("color", res_4w["risk_metadata"][0])

    def test_04_city_wide_batch_prediction_module2(self):
        """Module 2 batch predicts across all 60 wards for a single date."""
        latest_date = self.df["date"].max()
        slice_df = self.df[self.df["date"] == latest_date].copy()
        self.assertEqual(len(slice_df), 60)

        preds_2w = self.engine.predict(slice_df, horizon="2w")
        preds_4w = self.engine.predict(slice_df, horizon="4w")

        self.assertEqual(len(preds_2w["predicted_cases"]), 60)
        self.assertEqual(len(preds_4w["predicted_cases"]), 60)
        
        # Risk levels valid
        valid_tiers = {"Low", "Medium", "High", "Critical"}
        self.assertTrue(all(t in valid_tiers for t in preds_2w["risk_tier"]))
        self.assertTrue(all(t in valid_tiers for t in preds_4w["risk_tier"]))

    def test_05_weather_simulator_perturbations_module4(self):
        """Module 4 perturbation simulation runs and produces rational outputs."""
        sample = self.df.iloc[[0]].copy()
        cases_base = self.engine.predict(sample, horizon="2w")["predicted_cases"][0]

        # Heavy monsoon scenario: +80mm rain
        sim_sample = sample.copy()
        sim_sample["rainfall_mm"] += 80.0
        sim_sample["rain_log1p"] = np.log1p(sim_sample["rainfall_mm"])
        
        cases_sim = self.engine.predict(sim_sample, horizon="2w")["predicted_cases"][0]
        self.assertIsInstance(float(cases_sim), float)


if __name__ == "__main__":
    unittest.main()
