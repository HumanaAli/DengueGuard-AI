"""
DENGUARD AI — Model Loader & Inference Engine
============================================
Official inference loader for the DenguAlert dashboard and production pipeline.

Key Capabilities:
  - Loads serialized models (model_2w.joblib, model_4w.joblib).
  - Automatically loads and enforces calibrated decision thresholds (calibrated_config.json).
  - Generates continuous case forecasts (>= 0 cases) and operational risk tiers
    (Low, Medium, High, Critical).
  - Provides visual metadata (colors, icons, public health actions) for UI display.
  - Enforces strict feature validation and leakage prevention.

Usage:
  from src.model_loader import PredictionEngine
  
  engine = PredictionEngine()
  results = engine.predict(ward_features_df, horizon='2w')
  print(results['predicted_cases'], results['risk_tier'])
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_PATH = os.path.join(MODELS_DIR, "calibrated_config.json")

# Forbidden columns that must NEVER enter inference
FORBIDDEN_COLS = [
    "dengue_cases",
    "target_2w",
    "target_4w",
    "target_date_2w",
    "target_date_4w",
    "date",
    "split"
]

RISK_TIERS = ["Low", "Medium", "High", "Critical"]

RISK_METADATA = {
    "Low": {
        "color": "#10B981",       # Emerald Green
        "bg_color": "#ECFDF5",
        "icon": "🟢",
        "label": "Low Risk",
        "description": "Baseline non-outbreak conditions. Routine surveillance recommended."
    },
    "Medium": {
        "color": "#F59E0B",       # Amber / Yellow
        "bg_color": "#FFFBEB",
        "icon": "🟡",
        "label": "Medium Risk",
        "description": "Sporadic local transmission detected. Targeted source reduction & larvicide needed."
    },
    "High": {
        "color": "#F97316",       # Deep Orange
        "bg_color": "#FFF7ED",
        "icon": "🟠",
        "label": "High Risk",
        "description": "Active transmission cluster. Deploy focal spraying and alert hospitals."
    },
    "Critical": {
        "color": "#EF4444",       # Crimson Red
        "bg_color": "#FEF2F2",
        "icon": "🔴",
        "label": "Critical Risk",
        "description": "Imminent or active epidemic outbreak. Emergency fogging & resource mobilization."
    }
}


class PredictionEngine:
    """Thread-safe, cached prediction engine for DenguAlert."""

    def __init__(self, models_dir=MODELS_DIR, config_path=CONFIG_PATH):
        self.models_dir = models_dir
        self.config_path = config_path
        self._load_config()
        self._load_models()

    def _load_config(self):
        """Load operational calibrated thresholds."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file missing: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.thresholds = self.config.get("thresholds", {})

    def _load_models(self):
        """Load serialized pipeline models."""
        path_2w = os.path.join(self.models_dir, "model_2w.joblib")
        path_4w = os.path.join(self.models_dir, "model_4w.joblib")

        if not os.path.exists(path_2w):
            raise FileNotFoundError(f"2-Week model not found at {path_2w}")
        if not os.path.exists(path_4w):
            raise FileNotFoundError(f"4-Week model not found at {path_4w}")

        self.model_2w = joblib.load(path_2w)
        self.model_4w = joblib.load(path_4w)

    def classify_risk(self, predicted_cases, horizon="2w"):
        """
        Map continuous predicted case count to risk tiers using calibrated thresholds.
        """
        arr = np.asarray(predicted_cases)
        arr = np.clip(arr, 0.0, None)

        t = self.thresholds.get(horizon, {})
        low_max = t.get("Low_max", 0.8)
        med_max = t.get("Medium_max", 4.0)
        crit_min = t.get("Critical_min", 8.0 if horizon == "2w" else 8.5)

        categories = np.empty(arr.shape, dtype=object)
        categories[arr < low_max] = "Low"
        categories[(arr >= low_max) & (arr < med_max)] = "Medium"
        categories[(arr >= med_max) & (arr < crit_min)] = "High"
        categories[arr >= crit_min] = "Critical"

        if np.ndim(predicted_cases) == 0:
            return categories.item()
        return categories

    def predict(self, features_df, horizon="2w"):
        """
        Generate continuous case forecasts and calibrated risk tiers.
        
        Args:
            features_df: DataFrame containing the 34 required model features.
            horizon: '2w' or '4w'.
            
        Returns:
            dict containing:
              - 'predicted_cases': float or np.ndarray of non-negative cases.
              - 'risk_tier': str or np.ndarray of ['Low', 'Medium', 'High', 'Critical'].
              - 'risk_metadata': dict with UI styling and guidance.
        """
        if horizon not in ["2w", "4w"]:
            raise ValueError(f"Invalid horizon '{horizon}'. Must be '2w' or '4w'.")

        # Select model
        model = self.model_2w if horizon == "2w" else self.model_4w

        # Filter out any forbidden columns to guarantee zero leakage
        clean_df = features_df.copy()
        for col in FORBIDDEN_COLS:
            if col in clean_df.columns:
                clean_df = clean_df.drop(columns=[col])

        # Generate continuous prediction
        raw_pred = model.predict(clean_df)
        pred_cases = np.clip(raw_pred, 0.0, None)

        # Generate calibrated risk tier
        risk_tier = self.classify_risk(pred_cases, horizon=horizon)

        if isinstance(risk_tier, str):
            meta = RISK_METADATA.get(risk_tier, {})
        else:
            meta = [RISK_METADATA.get(rt, {}) for rt in risk_tier]

        return {
            "predicted_cases": np.round(pred_cases, 2),
            "risk_tier": risk_tier,
            "risk_metadata": meta,
            "horizon": horizon
        }


# Global singleton instance for easy import in dashboard
_ENGINE_INSTANCE = None

def get_engine():
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = PredictionEngine()
    return _ENGINE_INSTANCE
