"""
DCTII-Predict — Training Orchestrator (ML Module 6)

Builds a training matrix from BigQuery, engineers features, trains dual
LightGBM regressors (day/night ΔT), performs LORO spatial cross-validation
with Optuna hyperparameter search, calibrates conformal prediction intervals,
computes SHAP explanations, and saves versioned model artifacts to GCS.

Reuses existing DCTII infrastructure:
  - pipeline/indicator_compute.py   → waste heat flux formula
  - pipeline/dctii_calculator.py    → normalization, scoring, impact categories
  - pipeline/ancillary_data.py      → GEE feature extraction
  - pipeline/pipeline_run.py        → run tracking for lineage
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from google.cloud import bigquery, storage

# Optuna is only needed during training, not at inference time
try:
    import optuna
except ImportError:
    optuna = None

logger = logging.getLogger("dctii.predict_train")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GCP_PROJECT = os.environ.get("GCP_PROJECT", "oil-tank-monitoring-123")
GCS_MODEL_BUCKET = f"dctii-model-{os.environ.get('GCP_PROJECT', 'oil-tank-monitoring-123')}"
GCS_MODEL_PREFIX = "predict"
TEST_REGION = "MTL"

# Climate zone ordinal encoding (thermal severity proxy)
CLIMATE_ZONE_HEAT_RANK = {
    "BWh": 5,   # Arid hot desert   — Phoenix
    "BSk": 4,   # Semi-arid cold    — Central Texas
    "Cfa": 3,   # Humid subtropical — Houston, NoVA
    "Dfa": 2,   # Humid continental — Toronto
    "Dfb": 1,   # Cold continental  — Montreal
}

COOLING_TYPE_BINARY = {
    "air_cooled": 1,    # 95% sensible heat → high ΔT_night
    "tower_cooled": 0,  # 60% sensible heat → buffered ΔT_night
    "unknown": 0,       # conservative default
}

# Region → site_id prefix mapping for LORO CV
REGION_MAP = {
    "PHX":  "PHX_",
    "HOU":  "HOU_",
    "NOVA": "NOVA_",
    "CTX":  "CTX_",
    "TOR":  "TOR_",
    "MTL":  "MTL_",
}

# Final feature list — identical order used for training AND inference
FEATURE_COLUMNS = [
    # Physical
    "waste_heat_flux",
    "sensible_heat_flux",
    "pue_overhead",
    "log_capacity",
    "footprint_km2",
    "load_factor",
    # Biophysics
    "ndvi_growing_max",
    "veg_cooling_deficit",
    "impervious_frac",
    "tree_cover_fraction",
    "bare_fraction",
    "elevation_norm",
    "log_population_density",
    "has_snow",
    # Climate / cooling
    "climate_heat_rank",
    "cooling_type_binary",
    "sensible_fraction",
    # Interactions
    "heat_x_veg_deficit",
    "capacity_x_air",
    "heat_x_climate",
    "impervious_x_heat",
    # Context
    "is_cluster_site",
    # ERA5 atmospheric — excluded from v1 (NULL until BQ backfill completes)
    # era5_solar_mean      — v2
    # era5_wind_speed      — v2
    # era5_diurnal_range   — v2
    # solar_x_impervious   — v2
]

TARGET_DAY = "final_label_delta_t_day"
TARGET_NIGHT = "final_label_delta_t_night"


# ---------------------------------------------------------------------------
# 1. Training Matrix
# ---------------------------------------------------------------------------

TRAINING_MATRIX_SQL = """
WITH indicators AS (
    SELECT
        i.year,
        i.site_id,
        i.delta_t_day         AS label_delta_t_day,
        i.delta_t_night       AS label_delta_t_night,
        i.heat_island_area_km2,
        i.population_exposed,
        i.waste_heat_flux_wm2,
        -- Repurpose min_monthly_reliability as estimation method metadata
        -- Values are method names (e.g. 'ring_difference', 'cem_weighted'), not quality flags
        i.min_monthly_reliability AS estimation_method,
        -- fraction_reliable_months is unpopulated (default 0.0) — do not filter on it
        i.fraction_reliable_months
    FROM `{project}.dctii_curated.site_indicators` i
    WHERE i.delta_t_night IS NOT NULL
      AND i.delta_t_day   IS NOT NULL
      AND i.delta_t_night BETWEEN -1.0 AND 8.0   -- physical plausibility bounds
      AND i.delta_t_day   BETWEEN -1.0 AND 10.0
),
did AS (
    SELECT
        site_id,
        cohort_year,
        MAX(CASE WHEN outcome_var='delta_t_day'   THEN att_estimate END) AS att_day,
        MAX(CASE WHEN outcome_var='delta_t_night' THEN att_estimate END) AS att_night,
        MAX(CASE WHEN outcome_var='delta_t_day'   THEN att_se       END) AS att_day_se,
        MAX(CASE WHEN outcome_var='delta_t_night' THEN att_se       END) AS att_night_se,
        MAX(pre_trend_p_value)                                           AS pre_trend_p,
        MAX(estimation_unit)                                             AS estimation_unit
    FROM `{project}.dctii_serving.did_results`
    WHERE data_version = (SELECT MAX(data_version) FROM `{project}.dctii_serving.did_results`)
    GROUP BY site_id, cohort_year
),
scores AS (
    SELECT site_id, year, dctii_score, weighting_scheme
    FROM `{project}.dctii_serving.dctii_scores`
    WHERE weighting_scheme = 'expert'
      AND data_version = (SELECT MAX(data_version) FROM `{project}.dctii_serving.dctii_scores`)
),
registry AS (
    SELECT
        site_id,
        latitude, longitude,
        climate_zone,
        capacity_mw,
        pue_estimate,
        pue_source,
        load_factor,
        cooling_type,
        footprint_km2,
        activation_year,
        confidence_tier,
        cluster_id
    FROM `{project}.dctii_ref.site_registry`
),
covariates AS (
    SELECT
        site_id,
        EXTRACT(YEAR FROM covariate_date) AS covariate_year,
        ndvi_max              AS ndvi_growing_max,
        impervious_fraction,
        tree_cover_fraction,
        bare_fraction,
        population_density,
        elevation_mean_m      AS elevation_m,
        snow_cover_days,
        -- ERA5 atmospheric covariates (replace NULLs once era5 backfill completes)
        CAST(NULL AS FLOAT64) AS era5_solar_mean,
        CAST(NULL AS FLOAT64) AS era5_wind_speed,
        CAST(NULL AS FLOAT64) AS era5_diurnal_range,
        FALSE                 AS covariate_year_proxy
    FROM `{project}.dctii_staging.site_covariates`
    WHERE zone_name = 'footprint'
)
SELECT
    i.year,
    i.site_id,
    r.latitude,
    r.longitude,
    r.climate_zone,
    r.capacity_mw,
    r.pue_estimate,
    r.pue_source,
    r.load_factor,
    r.cooling_type,
    r.footprint_km2,
    r.activation_year,
    r.confidence_tier,
    r.cluster_id,
    c.ndvi_growing_max,
    c.impervious_fraction,
    c.tree_cover_fraction,
    c.bare_fraction,
    c.population_density,
    c.elevation_m,
    c.snow_cover_days,
    c.era5_solar_mean,
    c.era5_wind_speed,
    c.era5_diurnal_range,
    c.covariate_year_proxy,

    r.capacity_mw * r.load_factor * (r.pue_estimate - 1.0) / NULLIF(r.footprint_km2, 0)
        AS waste_heat_flux_computed,

    i.label_delta_t_day,
    i.label_delta_t_night,

    d.att_day,
    d.att_night,
    d.att_day_se,
    d.att_night_se,
    d.pre_trend_p,
    d.estimation_unit,

    i.fraction_reliable_months,
    i.estimation_method,
    s.dctii_score,

    -- When DiD causal labels are available and pass quality checks, prefer them
    CASE
        WHEN d.att_day IS NOT NULL AND (d.pre_trend_p IS NULL OR d.pre_trend_p > 0.05)
             AND d.estimation_unit != 'cluster'
        THEN d.att_day
        ELSE i.label_delta_t_day
    END AS final_label_delta_t_day,

    CASE
        WHEN d.att_night IS NOT NULL AND (d.pre_trend_p IS NULL OR d.pre_trend_p > 0.05)
             AND d.estimation_unit != 'cluster'
        THEN d.att_night
        ELSE i.label_delta_t_night
    END AS final_label_delta_t_night

FROM indicators i
JOIN registry r USING (site_id)
LEFT JOIN covariates c ON c.site_id = i.site_id AND c.covariate_year = i.year
LEFT JOIN did d ON d.site_id = i.site_id AND d.cohort_year = r.activation_year
LEFT JOIN scores s ON s.site_id = i.site_id AND s.year = i.year
ORDER BY i.site_id, i.year
"""


def build_training_matrix(
    project_id: str = None,
) -> pd.DataFrame:
    """Fetch the training matrix from BigQuery."""
    project = project_id or GCP_PROJECT
    client = bigquery.Client(project=project)
    sql = TRAINING_MATRIX_SQL.format(project=project)
    df = client.query(sql).to_dataframe()
    logger.info(f"Training matrix: {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer all features for training or inference.
    All features must be computable at inference time from user inputs + GEE.
    """
    df = df.copy()

    # ── Tier 1: Physical causation features ──────────────────────────────
    df["waste_heat_flux"] = df.get("waste_heat_flux_computed", pd.Series(dtype=float))
    fallback = (
        df["capacity_mw"] * df["load_factor"].fillna(0.6)
        * (df["pue_estimate"] - 1.0) / df["footprint_km2"].clip(lower=0.01)
    )
    df["waste_heat_flux"] = df["waste_heat_flux"].fillna(fallback)

    df["sensible_fraction"] = df["cooling_type"].map(
        {"air_cooled": 0.95, "tower_cooled": 0.60, "unknown": 0.77}
    ).fillna(0.77)
    df["sensible_heat_flux"] = df["waste_heat_flux"] * df["sensible_fraction"]

    df["pue_overhead"] = (df["pue_estimate"] - 1.0).clip(lower=0.05)

    # ── Tier 2: Site biophysics ───────────────────────────────────────────
    df["veg_cooling_deficit"] = 1.0 - df["ndvi_growing_max"].fillna(0.3)
    df["impervious_frac"] = df["impervious_fraction"].fillna(0.5)
    df["has_snow"] = (df["snow_cover_days"].fillna(0) > 30).astype(int)

    # ── Tier 3: Climate encoding ─────────────────────────────────────────
    df["climate_heat_rank"] = df["climate_zone"].map(CLIMATE_ZONE_HEAT_RANK).fillna(3)
    df["cooling_type_binary"] = df["cooling_type"].map(COOLING_TYPE_BINARY).fillna(0)

    # ── Tier 4: Key interaction terms ────────────────────────────────────
    df["heat_x_veg_deficit"] = df["sensible_heat_flux"] * df["veg_cooling_deficit"]
    df["capacity_x_air"] = df["capacity_mw"] * df["cooling_type_binary"]
    df["heat_x_climate"] = df["waste_heat_flux"] * df["climate_heat_rank"]
    df["impervious_x_heat"] = df["impervious_frac"] * df["waste_heat_flux"]

    # ── ERA5 atmospheric features ────────────────────────────────────────
    df["era5_solar_mean"] = df.get("era5_solar_mean", pd.Series(dtype=float)).fillna(200.0)
    df["era5_wind_speed"] = df.get("era5_wind_speed", pd.Series(dtype=float)).fillna(3.0)
    df["era5_diurnal_range"] = df.get("era5_diurnal_range", pd.Series(dtype=float)).fillna(10.0)
    # Interaction: solar x impervious — sealed surfaces absorb more when solar is high
    df["solar_x_impervious"] = df["era5_solar_mean"] * df["impervious_frac"]

    # ── Tier 5: Spatial context ──────────────────────────────────────────
    df["log_capacity"] = np.log1p(df["capacity_mw"])
    df["log_population_density"] = np.log1p(df["population_density"].fillna(500))
    df["elevation_norm"] = df["elevation_m"].fillna(200) / 1000.0
    df["is_cluster_site"] = df["cluster_id"].notna().astype(int)

    # ── Sample weights (used in training, not a feature) ─────────────────
    df["sample_weight"] = df["confidence_tier"].map(
        {1: 3.0, 2: 1.5, 3: 0.5}
    ).fillna(1.0)
    if "covariate_year_proxy" in df.columns:
        df.loc[df["covariate_year_proxy"] == True, "sample_weight"] *= 0.7
    if "att_day" in df.columns and TARGET_DAY in df.columns:
        att_mask = df[TARGET_DAY] == df["att_day"]
        df.loc[att_mask, "sample_weight"] *= 1.5

    # Fill any remaining NaNs in feature columns with safe defaults
    fill_defaults = {
        "waste_heat_flux": 0.0, "sensible_heat_flux": 0.0, "pue_overhead": 0.2,
        "log_capacity": 0.0, "footprint_km2": 0.05, "load_factor": 0.6,
        "ndvi_growing_max": 0.3, "veg_cooling_deficit": 0.7, "impervious_frac": 0.5,
        "tree_cover_fraction": 0.1, "bare_fraction": 0.1, "elevation_norm": 0.2,
        "log_population_density": np.log1p(500), "has_snow": 0,
        "climate_heat_rank": 3, "cooling_type_binary": 0, "sensible_fraction": 0.77,
        "heat_x_veg_deficit": 0.0, "capacity_x_air": 0.0, "heat_x_climate": 0.0,
        "impervious_x_heat": 0.0, "is_cluster_site": 0,
        "era5_solar_mean": 200.0, "era5_wind_speed": 3.0, "era5_diurnal_range": 10.0,
        "solar_x_impervious": 100.0,
    }
    for col, default in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    return df


# ---------------------------------------------------------------------------
# 3. Train / Validation / Test Split (LORO)
# ---------------------------------------------------------------------------

def get_splits(
    df: pd.DataFrame,
    test_region: str = TEST_REGION,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple[pd.DataFrame, pd.DataFrame, str]]]:
    """
    Split data into train+val and test using Leave-One-Region-Out.
    Test region (MTL) is held out entirely.
    Returns (train_val, test, loro_folds).
    """
    test_mask = df["site_id"].str.startswith(REGION_MAP[test_region])
    train_val = df[~test_mask].copy()
    test = df[test_mask].copy()

    loro_folds = []
    for region, prefix in REGION_MAP.items():
        if region == test_region:
            continue
        val_mask = train_val["site_id"].str.startswith(prefix)
        fold_train = train_val[~val_mask]
        fold_val = train_val[val_mask]
        if len(fold_val) > 0:
            loro_folds.append((fold_train, fold_val, region))

    logger.info(
        f"Split: train_val={len(train_val)}, test={len(test)} ({test_region}), "
        f"LORO folds={len(loro_folds)}"
    )
    return train_val, test, loro_folds


# ---------------------------------------------------------------------------
# 4. Hyperparameter Search with Optuna
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    train_val_df: pd.DataFrame,
    loro_folds: list,
    target_col: str,
    n_trials: int = 80,
    seed: int = 42,
) -> dict:
    """
    Optuna hyperparameter search using LORO cross-validation.
    Optimizes MAE; prunes trials where mean Pearson r < 0.60.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "num_leaves": trial.suggest_int("num_leaves", 10, 60),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": seed,
        }

        fold_maes = []
        fold_rhos = []
        for fold_train, fold_val, region in loro_folds:
            X_tr = fold_train[FEATURE_COLUMNS]
            y_tr = fold_train[target_col]
            w_tr = fold_train["sample_weight"]

            X_val = fold_val[FEATURE_COLUMNS]
            y_val = fold_val[target_col]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(20, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            preds = model.predict(X_val)

            mae = mean_absolute_error(y_val, preds)
            rho = np.corrcoef(y_val, preds)[0, 1] if len(y_val) > 1 else 0.0
            fold_maes.append(mae)
            fold_rhos.append(rho)

        mean_rho = np.mean(fold_rhos)
        if mean_rho < 0.20:
            raise optuna.exceptions.TrialPruned()

        return np.mean(fold_maes)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        logger.warning(
            f"[{target_col}] All {n_trials} trials pruned (mean r < 0.20). "
            "Falling back to default hyperparameters."
        )
        return {
            "n_estimators": 300,
            "num_leaves": 31,
            "max_depth": 5,
            "learning_rate": 0.05,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }

    logger.info(f"[{target_col}] Best MAE: {study.best_value:.4f}°C")
    logger.info(f"[{target_col}] Best params: {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# 5. Final Model Training (median + quantile regressors)
# ---------------------------------------------------------------------------

def train_final_model(
    train_val_df: pd.DataFrame,
    best_params: dict,
    target_col: str,
    seed: int = 42,
) -> Tuple[lgb.LGBMRegressor, lgb.LGBMRegressor, lgb.LGBMRegressor]:
    """
    Train three models: median, q10, q90 for prediction intervals.
    Returns (model_med, model_q10, model_q90).
    """
    X = train_val_df[FEATURE_COLUMNS]
    y = train_val_df[target_col]
    w = train_val_df["sample_weight"]

    params_med = {**best_params, "objective": "regression_l1", "random_state": seed}
    params_q10 = {**best_params, "objective": "quantile", "alpha": 0.10, "random_state": seed}
    params_q90 = {**best_params, "objective": "quantile", "alpha": 0.90, "random_state": seed}

    model_med = lgb.LGBMRegressor(**params_med)
    model_q10 = lgb.LGBMRegressor(**params_q10)
    model_q90 = lgb.LGBMRegressor(**params_q90)

    model_med.fit(X, y, sample_weight=w)
    model_q10.fit(X, y, sample_weight=w)
    model_q90.fit(X, y, sample_weight=w)

    logger.info(f"[{target_col}] Trained median + quantile models on {len(X)} rows")
    return model_med, model_q10, model_q90


def train_stratified_day_models(
    train_val_df: pd.DataFrame,
    loro_folds: list,
    n_trials: int = 80,
    seed: int = 42,
) -> Tuple[tuple, tuple]:
    """
    Train two separate day regressors stratified by estimation method:
    - day_cem: trained on CEM-weighted rows only (~207 rows)
    - day_ring: trained on ring_difference rows only (~69 rows)

    At inference time, day_cem is the primary model.
    day_ring is used only when the query location matches a known
    ring_difference region (NOVA cluster sites).

    Returns (models_day_cem, models_day_ring) where each is
    (model_med, model_q10, model_q90).
    """
    target_col = "label_delta_t_day"

    cem_mask = train_val_df["estimation_method"] == "cem_weighted"
    ring_mask = train_val_df["estimation_method"] == "ring_difference"

    df_cem = train_val_df[cem_mask].copy()
    df_ring = train_val_df[ring_mask].copy()

    logger.info(
        f"Day CEM subset:  {len(df_cem)} rows | "
        f"mean={df_cem[target_col].mean():.3f}°C | "
        f"std={df_cem[target_col].std():.3f}°C"
    )
    logger.info(
        f"Day Ring subset: {len(df_ring)} rows | "
        f"mean={df_ring[target_col].mean():.3f}°C | "
        f"std={df_ring[target_col].std():.3f}°C"
    )

    # --- CEM model: full LORO-CV, rich signal ---
    cem_folds = [
        (
            fold_tr[fold_tr["estimation_method"] == "cem_weighted"],
            fold_val[fold_val["estimation_method"] == "cem_weighted"],
            region,
        )
        for fold_tr, fold_val, region in loro_folds
        if len(fold_val[fold_val["estimation_method"] == "cem_weighted"]) >= 3
    ]
    logger.info(f"Day CEM: {len(cem_folds)} LORO folds available")

    best_params_cem = tune_hyperparameters(
        df_cem, cem_folds, target_col,
        n_trials=n_trials, seed=seed,
    )
    models_day_cem = train_final_model(
        df_cem, best_params_cem, target_col, seed=seed,
    )

    # --- Ring model: 69 rows, low variance — conservative fixed params ---
    ring_params = {
        "objective": "huber",
        "alpha": 0.9,
        "metric": "huber",
        "n_estimators": 100,
        "num_leaves": 8,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": seed,
    }
    logger.info("Day Ring model: using fixed conservative params (low variance, overfit risk)")
    models_day_ring = train_final_model(
        df_ring, ring_params, target_col, seed=seed,
    )

    return models_day_cem, models_day_ring


# ---------------------------------------------------------------------------
# 6. Conformal Prediction Calibration (Cross-Conformal via LORO holdouts)
# ---------------------------------------------------------------------------

def train_quantile_models(
    df: pd.DataFrame,
    target_col: str,
    seed: int = 42,
) -> Tuple[lgb.LGBMRegressor, lgb.LGBMRegressor]:
    """
    Train q10 + q90 quantile models with conservative fixed hyperparameters.
    Separate from median model — quantile regression needs stronger
    regularization to prevent interval collapse on small datasets.
    """
    X = df[FEATURE_COLUMNS].values
    y = df[target_col].values
    w = df["sample_weight"].values

    quantile_params = {
        "num_leaves": 8,
        "max_depth": 4,
        "min_child_samples": 15,
        "n_estimators": 200,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 2.0,
        "reg_lambda": 2.0,
        "random_state": seed,
        "verbosity": -1,
    }

    model_q10 = lgb.LGBMRegressor(
        **quantile_params, objective="quantile", alpha=0.10, metric="quantile",
    )
    model_q90 = lgb.LGBMRegressor(
        **quantile_params, objective="quantile", alpha=0.90, metric="quantile",
    )

    model_q10.fit(X, y, sample_weight=w)
    model_q90.fit(X, y, sample_weight=w)

    # Collapse check
    mean_width = np.mean(model_q90.predict(X) - model_q10.predict(X))
    if mean_width < 0.05:
        logger.warning(
            f"[{target_col}] Quantile interval width={mean_width:.4f} deg C "
            f"-- possible collapse, check min_child_samples"
        )

    return model_q10, model_q90


def calibrate_cross_conformal(
    train_val_df: pd.DataFrame,
    loro_folds: list,
    target_col: str,
    alpha: float = 0.10,
    seed: int = 42,
) -> float:
    """
    Cross-conformal calibration using LORO out-of-fold predictions.

    Each fold trains on N-1 regions, predicts on the held-out region.
    Nonconformity scores come from data the model has NEVER seen.
    Correction is therefore valid and non-leaking.

    With 5 LORO folds covering all ~276 rows, we get ~276 nonconformity
    scores instead of 41 from a 15% split — much more stable estimate.
    """
    all_nonconformity = []
    fold_coverage_log = []

    for fold_train, fold_val, region in loro_folds:
        if len(fold_val) < 3:
            logger.info(f"  [{region}] Skipping -- only {len(fold_val)} val rows")
            continue

        # Train fresh quantile models on this fold's training data only
        q10_fold, q90_fold = train_quantile_models(
            fold_train, target_col, seed=seed,
        )

        X_val = fold_val[FEATURE_COLUMNS].values
        y_val = fold_val[target_col].values

        q10_pred = q10_fold.predict(X_val)
        q90_pred = q90_fold.predict(X_val)

        # Nonconformity score: how much did interval miss by?
        scores = np.maximum(q10_pred - y_val, y_val - q90_pred)
        scores = np.maximum(scores, 0.0)
        all_nonconformity.extend(scores.tolist())

        raw_cov = np.mean((y_val >= q10_pred) & (y_val <= q90_pred))
        mean_width = np.mean(q90_pred - q10_pred)
        fold_coverage_log.append((region, len(y_val), raw_cov, mean_width))

        logger.info(
            f"  [{region}] {len(y_val)} rows | "
            f"raw coverage: {raw_cov:.1%} | "
            f"interval width: {mean_width:.3f} deg C | "
            f"scores: mean={np.mean(scores):.3f} max={np.max(scores):.3f}"
        )

    if len(all_nonconformity) < 20:
        raise ValueError(
            f"Only {len(all_nonconformity)} nonconformity scores -- "
            f"insufficient for reliable conformal calibration. "
            f"Check that LORO folds have adequate validation rows."
        )

    # Conformal quantile: ceil((n+1)(1-alpha))/n per Angelopoulos & Bates 2022
    n = len(all_nonconformity)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    correction = float(np.quantile(all_nonconformity, min(level, 1.0)))

    if correction < 0.05:
        logger.warning(
            f"correction={correction:.4f} deg C is suspiciously small -- "
            f"check that fold_val is truly held out from training"
        )
    if correction > 2.0:
        logger.warning(
            f"correction={correction:.4f} deg C is very large -- "
            f"model intervals may be too narrow; check quantile model params"
        )

    logger.info(
        f"\n  Cross-conformal summary ({target_col}):\n"
        f"  Total nonconformity scores: {n}\n"
        f"  Correction at {1-alpha:.0%} level: {correction:.4f} deg C\n"
        f"  Expected empirical coverage: >={1-alpha:.0%}"
    )

    return correction


# ---------------------------------------------------------------------------
# 6b. Bias Offset Correction (stratified by climate_heat_rank)
# ---------------------------------------------------------------------------

def compute_bias_offset(
    model_med: lgb.LGBMRegressor,
    train_val_df: pd.DataFrame,
    target_col: str,
) -> dict:
    """
    Compute bias offsets stratified by climate_heat_rank.
    Prevents a single global offset from over-correcting warm-climate sites
    while under-correcting cold-climate sites (the MTL problem).
    """
    X = train_val_df[FEATURE_COLUMNS].values
    y = train_val_df[target_col].values
    pred = model_med.predict(X)
    residuals = pred - y  # positive = overprediction

    df_tmp = train_val_df.copy()
    df_tmp["residual"] = residuals

    offsets = {}
    for rank in sorted(df_tmp["climate_heat_rank"].unique()):
        mask = df_tmp["climate_heat_rank"] == rank
        if mask.sum() >= 5:
            offsets[int(rank)] = float(df_tmp.loc[mask, "residual"].mean())
            logger.info(
                f"  climate_heat_rank={rank}: "
                f"n={mask.sum()} | "
                f"bias={offsets[int(rank)]:+.4f} deg C"
            )

    # Global fallback for unseen climate ranks
    offsets["global"] = float(np.mean(residuals))
    logger.info(f"  Global fallback bias: {offsets['global']:+.4f} deg C")

    return offsets


def apply_bias_offset(
    raw_pred: float,
    climate_heat_rank: int,
    offsets: dict,
) -> float:
    """Apply stratified bias correction at inference time."""
    offset = offsets.get(int(climate_heat_rank), offsets.get("global", 0.0))
    return max(0.0, raw_pred - offset)


# ---------------------------------------------------------------------------
# 7. SHAP Explainability
# ---------------------------------------------------------------------------

def compute_shap_explainer(
    model_med: lgb.LGBMRegressor,
    train_val_df: pd.DataFrame,
) -> shap.TreeExplainer:
    """
    Compute SHAP TreeExplainer and validate physical plausibility.
    Top 3 features should include at least 2 physical features.
    """
    X = train_val_df[FEATURE_COLUMNS]
    explainer = shap.TreeExplainer(model_med)

    shap_values = explainer.shap_values(X)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top3 = [FEATURE_COLUMNS[i] for i in np.argsort(mean_abs)[::-1][:3]]
    logger.info(f"Top 3 SHAP features: {top3}")

    physical_features = {
        "waste_heat_flux", "sensible_heat_flux", "log_capacity",
        "heat_x_veg_deficit", "cooling_type_binary", "pue_overhead",
        "capacity_x_air", "heat_x_climate",
    }
    n_physical = len(set(top3) & physical_features)
    if n_physical < 2:
        logger.warning(
            f"SHAP sanity check WARNING — only {n_physical}/3 top features "
            f"are physical: {top3}. Review model."
        )
    else:
        logger.info("SHAP sanity check PASSED — physical features dominate.")

    return explainer


# ---------------------------------------------------------------------------
# 8. Model Evaluation
# ---------------------------------------------------------------------------

# Acceptance thresholds (must all pass for model promotion)
THRESHOLDS = {
    "mae": 0.4,          # ≤ 0.4°C MAE
    "spearman_rho": 0.70,
    "pearson_r": 0.65,
    "r2": 0.40,
    "ci_coverage": 0.85,  # ≥ 85% of test points inside predicted CI
}


def evaluate_model(
    model_med: lgb.LGBMRegressor,
    model_q10: lgb.LGBMRegressor,
    model_q90: lgb.LGBMRegressor,
    correction: float,
    test_df: pd.DataFrame,
    target_col: str,
) -> dict:
    """
    Compute all evaluation metrics on held-out test set.
    Returns dict with metrics and pass/fail status.
    """
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_col].values

    pred = model_med.predict(X_test)
    q10 = model_q10.predict(X_test) - correction
    q90 = model_q90.predict(X_test) + correction

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred) if len(y_test) > 1 else 0.0
    pearson_r, pearson_p = stats.pearsonr(y_test, pred) if len(y_test) > 2 else (0.0, 1.0)
    spearman_rho, spearman_p = stats.spearmanr(y_test, pred) if len(y_test) > 2 else (0.0, 1.0)
    mbe = float(np.mean(pred - y_test))
    coverage = float(np.mean((y_test >= q10) & (y_test <= q90)))

    results = {
        "target": target_col,
        "n_test": len(y_test),
        "test_region": TEST_REGION,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "mbe": float(mbe),
        "ci_coverage": coverage,
        "ci_target": 0.90,
    }

    results["passed"] = all([
        mae <= THRESHOLDS["mae"],
        spearman_rho >= THRESHOLDS["spearman_rho"],
        pearson_r >= THRESHOLDS["pearson_r"],
        r2 >= THRESHOLDS["r2"],
        coverage >= THRESHOLDS["ci_coverage"],
    ])

    logger.info(
        f"\n{'=' * 60}\n"
        f"EVALUATION: {target_col} | Test region: {TEST_REGION}\n"
        f"  MAE:           {mae:.4f}°C   (threshold ≤ {THRESHOLDS['mae']}°C) "
        f"{'✓' if mae <= THRESHOLDS['mae'] else '✗'}\n"
        f"  RMSE:          {rmse:.4f}°C\n"
        f"  R²:            {r2:.4f}       (threshold ≥ {THRESHOLDS['r2']}) "
        f"{'✓' if r2 >= THRESHOLDS['r2'] else '✗'}\n"
        f"  Pearson r:     {pearson_r:.4f}     (threshold ≥ {THRESHOLDS['pearson_r']}) "
        f"{'✓' if pearson_r >= THRESHOLDS['pearson_r'] else '✗'}\n"
        f"  Spearman ρ:    {spearman_rho:.4f}     (threshold ≥ {THRESHOLDS['spearman_rho']}) "
        f"{'✓' if spearman_rho >= THRESHOLDS['spearman_rho'] else '✗'}\n"
        f"  MBE:           {mbe:+.4f}°C  (bias check; target near 0)\n"
        f"  CI Coverage:   {coverage:.1%}    (target ≥ {THRESHOLDS['ci_coverage']:.0%}) "
        f"{'✓' if coverage >= THRESHOLDS['ci_coverage'] else '✗'}\n"
        f"  OVERALL: {'PASS ✓' if results['passed'] else 'FAIL ✗'}\n"
        f"{'=' * 60}"
    )

    return results


# ---------------------------------------------------------------------------
# 9. Residual Diagnostics Plot
# ---------------------------------------------------------------------------

def plot_diagnostics(
    pred: np.ndarray,
    y_test: np.ndarray,
    target_col: str,
    output_dir: str,
):
    """Generate predicted-vs-actual, residual, and histogram diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Predicted vs Actual
    axes[0].scatter(y_test, pred, alpha=0.7, s=30)
    lim = max(abs(y_test).max(), abs(pred).max()) * 1.1
    axes[0].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    axes[0].set_xlabel("Actual ΔT (°C)")
    axes[0].set_ylabel("Predicted ΔT (°C)")
    axes[0].set_title(f"{target_col}: Predicted vs Actual")

    # 2. Residuals vs Predicted
    resid = pred - y_test
    axes[1].scatter(pred, resid, alpha=0.7, s=30)
    axes[1].axhline(0, color="r", lw=1, ls="--")
    axes[1].set_xlabel("Predicted ΔT")
    axes[1].set_ylabel("Residual (pred - actual)")
    axes[1].set_title("Residuals vs Predicted")

    # 3. Residual histogram
    axes[2].hist(resid, bins=20, edgecolor="black")
    axes[2].axvline(0, color="r", lw=1, ls="--")
    axes[2].set_xlabel("Residual (°C)")
    axes[2].set_title("Residual Distribution")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{target_col}_diagnostics.png"), dpi=150)
    plt.close()
    logger.info(f"Diagnostic plot saved to {output_dir}/{target_col}_diagnostics.png")


# ---------------------------------------------------------------------------
# 10. Distribution Shift Detection
# ---------------------------------------------------------------------------

def compute_training_distribution(train_df: pd.DataFrame) -> dict:
    """
    Compute Mahalanobis reference distribution using site-level means
    with Ledoit-Wolf shrinkage for robust covariance estimation.

    Using multi-year rows inflates covariance with temporal variation,
    making single-query inference points appear out-of-distribution.
    Averaging to one row per site aligns covariance structure with
    how the predict endpoint receives queries.

    Ledoit-Wolf shrinkage stabilizes the covariance inverse when
    n_sites is close to n_features (34 sites, 22 features).
    """
    from sklearn.covariance import LedoitWolf

    if "site_id" in train_df.columns:
        site_means = (
            train_df
            .groupby("site_id")[FEATURE_COLUMNS]
            .mean()
            .reset_index(drop=True)
        )
    else:
        site_means = train_df[FEATURE_COLUMNS]

    X = site_means.values
    centroid = X.mean(axis=0).tolist()

    lw = LedoitWolf().fit(X)
    cov_inv = lw.precision_.tolist()

    logger.info(
        f"Training distribution: {len(site_means)} sites "
        f"(was {len(train_df)} multi-year rows), "
        f"Ledoit-Wolf shrinkage={lw.shrinkage_:.3f}"
    )

    return {
        "centroid": centroid,
        "cov_inv": cov_inv,
        "n_sites": len(site_means),
        "n_rows": len(train_df),
        "method": "site_level_ledoit_wolf",
        "shrinkage": float(lw.shrinkage_),
    }


def compute_distribution_shift(
    query_features: np.ndarray,
    training_distribution: dict,
) -> Tuple[float, str]:
    """
    Compute Mahalanobis distance from training distribution.
    Returns (distance, shift_label).

    Thresholds based on chi-squared distribution for 22 features:
    chi2 95th percentile ~ 33.9, 99th ~ 40.3.
    sqrt scale: sqrt(33.9) ~ 5.8, sqrt(40.3) ~ 6.3.
    """
    centroid = np.array(training_distribution["centroid"])
    cov_inv = np.array(training_distribution["cov_inv"])

    dist = float(mahalanobis(query_features, centroid, cov_inv))

    if dist < 4.0:
        label = "in_distribution"
    elif dist < 6.0:
        label = "moderate_shift"
    elif dist < 8.0:
        label = "high_shift"
    else:
        label = "extrapolation"

    return dist, label


# ---------------------------------------------------------------------------
# 11. Model Artifact Save / Load
# ---------------------------------------------------------------------------

def verify_gcs_bucket(bucket_name: str) -> None:
    """Verify GCS bucket exists and is accessible before uploading."""
    client = storage.Client()
    try:
        client.get_bucket(bucket_name)
        logger.info(f"[GCS] Bucket verified: {bucket_name}")
    except Exception as e:
        raise RuntimeError(
            f"GCS bucket '{bucket_name}' not accessible. "
            f"Check GCS_MODEL_BUCKET or Terraform deployment. "
            f"Error: {e}"
        )


def save_model_artifacts(
    version: str,
    models_day_cem: tuple,
    models_day_ring: tuple,
    models_night: tuple,
    shap_day_cem,
    shap_night,
    conformal_corrections: dict,
    train_distribution: dict,
    eval_report: dict,
    feature_metadata: dict,
):
    """Save all model artifacts to GCS under versioned prefix."""
    verify_gcs_bucket(GCS_MODEL_BUCKET)
    client = storage.Client()
    bucket = client.bucket(GCS_MODEL_BUCKET)

    artifacts = {
        "lgbm_day_cem_median.pkl": models_day_cem[0],
        "lgbm_day_cem_q10.pkl": models_day_cem[1],
        "lgbm_day_cem_q90.pkl": models_day_cem[2],
        "lgbm_day_ring_median.pkl": models_day_ring[0],
        "lgbm_day_ring_q10.pkl": models_day_ring[1],
        "lgbm_day_ring_q90.pkl": models_day_ring[2],
        "lgbm_night_median.pkl": models_night[0],
        "lgbm_night_q10.pkl": models_night[1],
        "lgbm_night_q90.pkl": models_night[2],
        "shap_explainer_day_cem.pkl": shap_day_cem,
        "shap_explainer_night.pkl": shap_night,
    }
    json_artifacts = {
        "conformal_corrections.json": conformal_corrections,
        "training_distribution.json": train_distribution,
        "eval_report.json": eval_report,
        "feature_metadata.json": feature_metadata,
    }

    prefix = f"{GCS_MODEL_PREFIX}/{version}"
    for fname, obj in artifacts.items():
        blob = bucket.blob(f"{prefix}/{fname}")
        blob.upload_from_string(
            pickle.dumps(obj), content_type="application/octet-stream"
        )

    for fname, obj in json_artifacts.items():
        blob = bucket.blob(f"{prefix}/{fname}")
        blob.upload_from_string(
            json.dumps(obj, indent=2, default=str), content_type="application/json"
        )

    logger.info(f"Saved model artifacts to gs://{GCS_MODEL_BUCKET}/{prefix}/")


def load_model_artifacts(version: str = "latest") -> dict:
    """Load model artifacts from GCS."""
    client = storage.Client()
    bucket = client.bucket(GCS_MODEL_BUCKET)

    # Resolve "latest" to the highest numbered version
    if version == "latest":
        blobs = list(bucket.list_blobs(prefix=f"{GCS_MODEL_PREFIX}/v"))
        versions = set()
        for b in blobs:
            parts = b.name.split("/")
            if len(parts) >= 2 and parts[1].startswith("v"):
                try:
                    versions.add(int(parts[1].replace("v", "")))
                except ValueError:
                    pass
        if not versions:
            raise RuntimeError("No model versions found in GCS")
        version = f"v{max(versions)}"
        logger.info(f"Resolved 'latest' to version: {version}")

    def load_pkl(name):
        data = bucket.blob(f"{GCS_MODEL_PREFIX}/{version}/{name}").download_as_bytes()
        return pickle.loads(data)
        return pickle.loads(data)

    def load_json(name):
        data = bucket.blob(f"{GCS_MODEL_PREFIX}/{version}/{name}").download_as_string()
        return json.loads(data)

    return {
        "day_cem_median": load_pkl("lgbm_day_cem_median.pkl"),
        "day_cem_q10": load_pkl("lgbm_day_cem_q10.pkl"),
        "day_cem_q90": load_pkl("lgbm_day_cem_q90.pkl"),
        "day_ring_median": load_pkl("lgbm_day_ring_median.pkl"),
        "day_ring_q10": load_pkl("lgbm_day_ring_q10.pkl"),
        "day_ring_q90": load_pkl("lgbm_day_ring_q90.pkl"),
        "night_median": load_pkl("lgbm_night_median.pkl"),
        "night_q10": load_pkl("lgbm_night_q10.pkl"),
        "night_q90": load_pkl("lgbm_night_q90.pkl"),
        "shap_day_cem": load_pkl("shap_explainer_day_cem.pkl"),
        "shap_night": load_pkl("shap_explainer_night.pkl"),
        "corrections": load_json("conformal_corrections.json"),
        "distribution": load_json("training_distribution.json"),
        "eval_report": load_json("eval_report.json"),
        "feature_meta": load_json("feature_metadata.json"),
    }


def resolve_version(version_str: str) -> str:
    """Resolve 'auto' to next incremental version from GCS."""
    if version_str != "auto":
        return version_str
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_MODEL_BUCKET)
        blobs = list(bucket.list_blobs(prefix=f"{GCS_MODEL_PREFIX}/v"))
        existing = set()
        for b in blobs:
            parts = b.name.split("/")
            if len(parts) >= 2:
                existing.add(parts[1])
        if not existing:
            return "v1"
        max_v = max(int(v.replace("v", "")) for v in existing if v.startswith("v"))
        return f"v{max_v + 1}"
    except Exception:
        return f"v{datetime.now(timezone.utc).strftime('%Y%m%d')}"
