"""
DCTII-Predict Unit Tests — no GEE, no BQ required.

Tests feature engineering, distribution shift detection,
conformal prediction, physical plausibility checks,
bias correction, DCTII scoring, and statistical utilities.
"""

import pytest
import numpy as np
import pandas as pd

from pipeline.predict_train import (
    engineer_features,
    FEATURE_COLUMNS,
    TARGET_DAY,
    TARGET_NIGHT,
    compute_training_distribution,
    compute_distribution_shift,
    apply_bias_offset,
    THRESHOLDS,
)
from pipeline.predict_infer import (
    resolve_pue,
    compose_dctii_score,
    get_climate_stratified_defaults,
    derive_day_from_night,
    spearman_confidence_interval,
)


# ── Mock Request ──────────────────────────────────────────────────────────

class MockRequest:
    """Minimal mock that mimics PredictRequest fields."""
    def __init__(self, pue=None, cooling_type="air_cooled",
                 capacity_mw=50.0, footprint_km2=None, load_factor=0.6):
        self.pue = pue
        self.cooling_type = cooling_type
        self.capacity_mw = capacity_mw
        self.footprint_km2 = footprint_km2
        self.load_factor = load_factor


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_row():
    return {
        "capacity_mw": 50.0,
        "pue_estimate": 1.25,
        "load_factor": 0.6,
        "cooling_type": "air_cooled",
        "footprint_km2": 0.1,
        "climate_zone": "Cfa",
        "cluster_id": None,
        "ndvi_growing_max": 0.35,
        "impervious_fraction": 0.7,
        "tree_cover_fraction": 0.05,
        "bare_fraction": 0.10,
        "population_density": 850.0,
        "elevation_m": 120.0,
        "snow_cover_days": 0.0,
        "covariate_year_proxy": False,
        "confidence_tier": 2,
    }


@pytest.fixture
def training_df(minimal_row):
    """Create a small synthetic training DataFrame."""
    rows = []
    for i in range(20):
        row = {
            **minimal_row,
            "site_id": f"TEST_{i:03d}",
            "year": 2020,
            "capacity_mw": 20 + i * 5,
            "ndvi_growing_max": 0.2 + i * 0.02,
            "label_delta_t_day": 0.5 + i * 0.1,
            "label_delta_t_night": 0.3 + i * 0.08,
            "att_day": None,
            "att_night": None,
            "waste_heat_flux_computed": None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ── Feature Engineering Tests ─────────────────────────────────────────────

def test_engineer_features_produces_all_columns(minimal_row):
    """All FEATURE_COLUMNS must be present after engineering."""
    df = engineer_features(pd.DataFrame([minimal_row]))
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature: {col}"


def test_no_nan_in_feature_columns(minimal_row):
    """Feature columns must have no NaN values after engineering."""
    df = engineer_features(pd.DataFrame([minimal_row]))
    X = df[FEATURE_COLUMNS]
    assert not X.isnull().any().any(), f"NaN in features: {X.isnull().sum().to_dict()}"


def test_waste_heat_flux_formula(minimal_row):
    """Waste heat flux = capacity * load_factor * (PUE-1) / footprint."""
    df = engineer_features(pd.DataFrame([minimal_row]))
    expected = 50.0 * 0.6 * (1.25 - 1.0) / 0.1  # = 75 W/m²
    assert abs(df["waste_heat_flux"].iloc[0] - expected) < 0.01


def test_air_cooled_higher_sensible_than_tower(minimal_row):
    """Air-cooled should produce higher sensible heat flux."""
    row_air = {**minimal_row, "cooling_type": "air_cooled"}
    row_tower = {**minimal_row, "cooling_type": "tower_cooled"}
    df_air = engineer_features(pd.DataFrame([row_air]))
    df_tower = engineer_features(pd.DataFrame([row_tower]))
    assert df_air["sensible_heat_flux"].iloc[0] > df_tower["sensible_heat_flux"].iloc[0]


def test_veg_cooling_deficit_bounded(minimal_row):
    """Vegetation cooling deficit must be in [0, 1]."""
    df = engineer_features(pd.DataFrame([minimal_row]))
    vcd = df["veg_cooling_deficit"].iloc[0]
    assert 0.0 <= vcd <= 1.0


def test_veg_cooling_deficit_high_for_low_ndvi(minimal_row):
    """Low NDVI = high vegetation cooling deficit."""
    row_low = {**minimal_row, "ndvi_growing_max": 0.1}
    row_high = {**minimal_row, "ndvi_growing_max": 0.8}
    df_low = engineer_features(pd.DataFrame([row_low]))
    df_high = engineer_features(pd.DataFrame([row_high]))
    assert df_low["veg_cooling_deficit"].iloc[0] > df_high["veg_cooling_deficit"].iloc[0]


def test_cluster_site_flag(minimal_row):
    """Cluster sites should have is_cluster_site=1."""
    row_cluster = {**minimal_row, "cluster_id": "NOVA_ASHBURN_CLUSTER"}
    df_cluster = engineer_features(pd.DataFrame([row_cluster]))
    df_isolated = engineer_features(pd.DataFrame([minimal_row]))
    assert df_cluster["is_cluster_site"].iloc[0] == 1
    assert df_isolated["is_cluster_site"].iloc[0] == 0


def test_snow_flag_threshold(minimal_row):
    """has_snow=1 when snow_cover_days > 30."""
    row_snow = {**minimal_row, "snow_cover_days": 60}
    row_nosnow = {**minimal_row, "snow_cover_days": 10}
    df_snow = engineer_features(pd.DataFrame([row_snow]))
    df_nosnow = engineer_features(pd.DataFrame([row_nosnow]))
    assert df_snow["has_snow"].iloc[0] == 1
    assert df_nosnow["has_snow"].iloc[0] == 0


def test_climate_heat_rank_ordering(minimal_row):
    """BWh (Phoenix) should rank higher than Dfb (Montreal)."""
    row_bwh = {**minimal_row, "climate_zone": "BWh"}
    row_dfb = {**minimal_row, "climate_zone": "Dfb"}
    df_bwh = engineer_features(pd.DataFrame([row_bwh]))
    df_dfb = engineer_features(pd.DataFrame([row_dfb]))
    assert df_bwh["climate_heat_rank"].iloc[0] > df_dfb["climate_heat_rank"].iloc[0]


def test_sample_weight_confidence_tier(minimal_row):
    """Tier-1 sites should get higher sample weights than tier-3."""
    row1 = {**minimal_row, "confidence_tier": 1}
    row3 = {**minimal_row, "confidence_tier": 3}
    df1 = engineer_features(pd.DataFrame([row1]))
    df3 = engineer_features(pd.DataFrame([row3]))
    assert df1["sample_weight"].iloc[0] > df3["sample_weight"].iloc[0]


def test_interaction_terms_computed(minimal_row):
    """Interaction terms should be non-zero for non-zero inputs."""
    df = engineer_features(pd.DataFrame([minimal_row]))
    assert df["heat_x_veg_deficit"].iloc[0] > 0
    assert df["heat_x_climate"].iloc[0] > 0
    assert df["impervious_x_heat"].iloc[0] > 0


def test_pue_overhead_clipped(minimal_row):
    """PUE overhead should be clipped to minimum 0.05."""
    row_low_pue = {**minimal_row, "pue_estimate": 1.01}
    df = engineer_features(pd.DataFrame([row_low_pue]))
    assert df["pue_overhead"].iloc[0] >= 0.05


# ── Distribution Shift Tests ──────────────────────────────────────────────

def test_distribution_shift_in_distribution():
    """A point at the centroid should be in_distribution."""
    np.random.seed(42)
    train = pd.DataFrame(
        np.random.randn(100, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS
    )
    dist = compute_training_distribution(train)
    centroid = np.array(dist["centroid"])
    score, label = compute_distribution_shift(centroid, dist)
    assert label == "in_distribution"
    assert score < 2.0


def test_distribution_shift_far_extrapolation():
    """A point far from training data should be labeled extrapolation."""
    np.random.seed(42)
    train = pd.DataFrame(
        np.random.randn(100, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS
    )
    dist = compute_training_distribution(train)
    far_point = np.ones(len(FEATURE_COLUMNS)) * 20
    score, label = compute_distribution_shift(far_point, dist)
    assert label == "extrapolation"
    assert score > 5.0


def test_distribution_shift_moderate():
    """A point somewhat away should be moderate_shift or higher."""
    np.random.seed(42)
    train = pd.DataFrame(
        np.random.randn(100, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS
    )
    dist = compute_training_distribution(train)
    moderate_point = np.array(dist["centroid"]) + 3.0
    score, label = compute_distribution_shift(moderate_point, dist)
    assert label in ("moderate_shift", "high_shift", "extrapolation")


# ── Feature Column Consistency ────────────────────────────────────────────

def test_feature_columns_count():
    """FEATURE_COLUMNS should have exactly 22 features."""
    assert len(FEATURE_COLUMNS) == 22


def test_feature_columns_no_duplicates():
    """No duplicate feature names."""
    assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))


# ── PUE Resolution Tests ─────────────────────────────────────────────────

def test_resolve_pue_uses_provided_value():
    """Explicit PUE is never overridden."""
    pue, imputed = resolve_pue(
        MockRequest(pue=1.35, cooling_type="air_cooled"), "BWh"
    )
    assert pue == 1.35
    assert imputed is False


def test_resolve_pue_imputes_when_none():
    """When PUE is None, imputation produces a valid PUE."""
    pue, imputed = resolve_pue(
        MockRequest(pue=None, cooling_type="air_cooled"), "BWh"
    )
    assert 1.0 < pue < 3.0
    assert imputed is True


# ── DCTII Score Tests ─────────────────────────────────────────────────────

def test_compose_dctii_score_bounds():
    """Score must always be 0-100 regardless of input extremes."""
    score, cat = compose_dctii_score(
        dt_day=99, dt_night=99, heat_area=99,
        pop_exposed=99999, waste_heat=9999, scheme="expert"
    )
    assert score == 100.0
    assert cat == "Severe"

    score, cat = compose_dctii_score(
        dt_day=0, dt_night=0, heat_area=0,
        pop_exposed=0, waste_heat=0, scheme="expert"
    )
    assert score == 0.0
    assert cat == "Minimal"


def test_compose_dctii_score_all_schemes():
    """All four weighting schemes return valid scores."""
    for scheme in ["expert", "equal", "pca", "entropy"]:
        score, cat = compose_dctii_score(
            dt_day=0.5, dt_night=0.3, heat_area=1.2,
            pop_exposed=5000, waste_heat=80, scheme=scheme
        )
        assert 0 <= score <= 100, f"scheme={scheme} produced score={score}"
        assert cat in ["Minimal", "Low", "Moderate", "High", "Severe"]


# ── Bias Offset Tests ────────────────────────────────────────────────────

def test_apply_bias_offset_respects_floor():
    """Physical floor: corrected ΔT cannot go below 0."""
    offsets = {2: 0.5, "global": 0.3}
    result = apply_bias_offset(raw_pred=0.1, climate_heat_rank=2, offsets=offsets)
    assert result == 0.0  # 0.1 - 0.5 = -0.4, floored to 0.0


# ── Day/Night Scaling Tests ──────────────────────────────────────────────

def test_day_night_scaling_all_climates():
    """All climate zones produce positive day ΔT from positive night ΔT."""
    for zone in ["BWh", "BSk", "Cfa", "Dfa", "Dfb"]:
        day = derive_day_from_night(dt_night=0.5, climate_zone=zone)
        assert day > 0.0, f"zone={zone} produced day={day}"
        assert day >= 0.5, f"zone={zone}: day should be >= night"


# ── Climate Fallback Tests ───────────────────────────────────────────────

def test_climate_fallback_has_all_required_keys():
    """GEE fallback must provide all keys needed by engineer_features()."""
    required = {
        "ndvi_growing_max", "impervious_fraction", "tree_cover_fraction",
        "bare_fraction", "population_density", "elevation_m", "snow_cover_days"
    }
    for zone in ["BWh", "BSk", "Cfa", "Dfa", "Dfb"]:
        defaults = get_climate_stratified_defaults(zone)
        missing = required - set(defaults.keys())
        assert not missing, f"zone={zone} missing keys: {missing}"


# ── Spearman CI Test ─────────────────────────────────────────────────────

def test_spearman_confidence_interval_contains_threshold():
    """Reproduce the MTL test set finding — documents the statistical decision."""
    lo, hi = spearman_confidence_interval(rho=0.630, n=26)
    assert lo < 0.70 < hi, "0.70 threshold should be inside CI for n=26, rho=0.63"
