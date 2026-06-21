"""
DCTII-Predict Golden / Regression Tests.

Run against known test cases (MTL test region sites) to catch model regressions.
These tests require a trained model to be available (mark as integration).
"""

import pytest
import numpy as np
import pandas as pd

from pipeline.predict_train import engineer_features, FEATURE_COLUMNS


# Golden test cases based on MTL region characteristics
GOLDEN_CASES = [
    {
        "name": "MTL_small_dc",
        "input": {
            "capacity_mw": 15.0, "pue_estimate": 1.35, "load_factor": 0.55,
            "cooling_type": "tower_cooled", "footprint_km2": 0.04,
            "climate_zone": "Dfb", "cluster_id": None,
            "ndvi_growing_max": 0.45, "impervious_fraction": 0.40,
            "tree_cover_fraction": 0.25, "bare_fraction": 0.05,
            "population_density": 720.0, "elevation_m": 35.0,
            "snow_cover_days": 90.0, "covariate_year_proxy": False,
            "confidence_tier": 2,
        },
        "expected_score_range": (2, 35),
    },
    {
        "name": "MTL_medium_dc",
        "input": {
            "capacity_mw": 40.0, "pue_estimate": 1.25, "load_factor": 0.6,
            "cooling_type": "air_cooled", "footprint_km2": 0.08,
            "climate_zone": "Dfb", "cluster_id": None,
            "ndvi_growing_max": 0.40, "impervious_fraction": 0.45,
            "tree_cover_fraction": 0.20, "bare_fraction": 0.05,
            "population_density": 750.0, "elevation_m": 30.0,
            "snow_cover_days": 85.0, "covariate_year_proxy": False,
            "confidence_tier": 2,
        },
        "expected_score_range": (10, 55),
    },
    {
        "name": "MTL_large_dc",
        "input": {
            "capacity_mw": 100.0, "pue_estimate": 1.20, "load_factor": 0.65,
            "cooling_type": "air_cooled", "footprint_km2": 0.17,
            "climate_zone": "Dfb", "cluster_id": None,
            "ndvi_growing_max": 0.35, "impervious_fraction": 0.55,
            "tree_cover_fraction": 0.15, "bare_fraction": 0.05,
            "population_density": 800.0, "elevation_m": 40.0,
            "snow_cover_days": 80.0, "covariate_year_proxy": False,
            "confidence_tier": 1,
        },
        "expected_score_range": (15, 70),
    },
]


class TestGoldenFeatureEngineering:
    """Verify feature engineering produces valid outputs for golden cases."""

    @pytest.mark.parametrize(
        "case", GOLDEN_CASES, ids=[c["name"] for c in GOLDEN_CASES]
    )
    def test_feature_columns_present(self, case):
        """All feature columns should be present."""
        df = engineer_features(pd.DataFrame([case["input"]]))
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"Missing: {col}"

    @pytest.mark.parametrize(
        "case", GOLDEN_CASES, ids=[c["name"] for c in GOLDEN_CASES]
    )
    def test_no_nan_features(self, case):
        """No NaN values in feature columns."""
        df = engineer_features(pd.DataFrame([case["input"]]))
        X = df[FEATURE_COLUMNS]
        assert not X.isnull().any().any()

    @pytest.mark.parametrize(
        "case", GOLDEN_CASES, ids=[c["name"] for c in GOLDEN_CASES]
    )
    def test_waste_heat_positive(self, case):
        """Waste heat flux must be positive for any active DC."""
        df = engineer_features(pd.DataFrame([case["input"]]))
        assert df["waste_heat_flux"].iloc[0] > 0

    @pytest.mark.parametrize(
        "case", GOLDEN_CASES, ids=[c["name"] for c in GOLDEN_CASES]
    )
    def test_snow_flag_for_montreal(self, case):
        """Montreal sites should have has_snow=1."""
        df = engineer_features(pd.DataFrame([case["input"]]))
        assert df["has_snow"].iloc[0] == 1, "MTL sites should have snow"


class TestGoldenPhysicalPlausibility:
    """Physical plausibility checks for golden cases."""

    def test_larger_dc_higher_waste_heat(self):
        """Larger DCs should produce more waste heat (all else equal)."""
        small = GOLDEN_CASES[0]["input"].copy()
        large = GOLDEN_CASES[2]["input"].copy()
        # Normalize to same PUE and load factor for fair comparison
        for row in (small, large):
            row["pue_estimate"] = 1.25
            row["load_factor"] = 0.6
        df_small = engineer_features(pd.DataFrame([small]))
        df_large = engineer_features(pd.DataFrame([large]))
        assert (
            df_large["waste_heat_flux"].iloc[0]
            > df_small["waste_heat_flux"].iloc[0]
        )

    def test_air_cooled_higher_sensible_fraction(self):
        """Air-cooled should have higher sensible fraction than tower."""
        air = {**GOLDEN_CASES[1]["input"], "cooling_type": "air_cooled"}
        tower = {**GOLDEN_CASES[1]["input"], "cooling_type": "tower_cooled"}
        df_air = engineer_features(pd.DataFrame([air]))
        df_tower = engineer_features(pd.DataFrame([tower]))
        assert (
            df_air["sensible_fraction"].iloc[0]
            > df_tower["sensible_fraction"].iloc[0]
        )
