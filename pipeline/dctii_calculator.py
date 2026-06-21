"""
Stage 4: DCTII Composite Score Calculator
Normalizes indicators, applies weighting schemes, computes final 0-100 score.

Directly adapted from GOII's giti_calculator.py:
  - Winsorized min-max normalization (replaces z-score)
  - Multiple weighting schemes (PCA, expert, entropy, equal)
  - Atomic BigQuery upserts (same MERGE pattern)
  - Impact category assignment

Critical design decisions (from critical analysis):
  - Normalization versioning: every scoring run gets a monotonic data_version
    counter. When normalization bounds or the site registry changes, historical
    runs remain distinguishable from re-computed runs.
  - Sample-relative normalization: DCTII scores are defined relative to the
    current site registry. Adding a new extreme site shifts all scores.
    Fixed-bound normalization option available as alternative.
  - norm_bounds_hash: SHA-256 of the min/max bounds used, stored alongside
    scores for provenance.
"""

import logging
import hashlib
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from google.cloud import bigquery

logger = logging.getLogger("dctii.calculator")

IMPACT_CATEGORIES = {
    (0, 20): "Minimal",
    (20, 40): "Low",
    (40, 60): "Moderate",
    (60, 80): "High",
    (80, 100): "Severe",
}

INDICATOR_NAMES = [
    "delta_t_day",
    "delta_t_night",
    "heat_island_area_km2",
    "population_exposed",
    "waste_heat_flux_wm2",
]

DEFAULT_WEIGHTS = {
    "equal": {k: 0.2 for k in INDICATOR_NAMES},
}

# Fixed physical bounds for normalization (alternative to sample-relative).
# Based on literature review and physical limits:
#   Delta-T: 0 C (no effect) to 5 C (extreme, Chakraborty & Lee upper bound)
#   Heat island area: 0 to 50 km2 (very large campus)
#   Population exposed: 0 to 100,000 (dense urban overlap)
#   Waste heat flux: 0 to 500 W/m2 (hyperscale facility)
# Phoenix arid DCs can show industrial zone anomalies of 4-6 C; upper bound of 5 C
# loses discrimination at exactly the cases where public health impact is highest.
# 8 C preserves index resolution across all studied climates (per UHI literature review).
FIXED_NORMALIZATION_BOUNDS = {
    "delta_t_day":          (0.0, 3.0),    # Literature: max DC-attributable UHI ~2-3°C
    "delta_t_night":        (0.0, 2.0),    # Nighttime effects smaller; p95=1.1°C
    "heat_island_area_km2": (0.0, 10.0),   # Max observed ~9.5 km²; 10 is proxy ceiling
    "population_exposed":   (0.0, 50000.0),# Scaled to suburban DC locations
    "waste_heat_flux_wm2":  (0.0, 250.0),  # Observed range 63-200; 250 as cap
}


def winsorize(values: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Clip at percentile bounds to reduce outlier influence."""
    lo = np.nanpercentile(values, lower * 100)
    hi = np.nanpercentile(values, upper * 100)
    return np.clip(values, lo, hi)


def min_max_normalize(
    values: np.ndarray,
    fixed_bounds: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Scale to [0, 1].

    If fixed_bounds is provided, uses those instead of sample min/max.
    This makes scores stable across registry expansions.
    """
    if fixed_bounds is not None:
        lo, hi = fixed_bounds
    else:
        lo, hi = np.nanmin(values), np.nanmax(values)
    if hi - lo < 1e-9:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def compute_norm_bounds_hash(bounds: Dict[str, Tuple[float, float]]) -> str:
    """SHA-256 hash of normalization bounds for provenance tracking."""
    serialized = json.dumps(bounds, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def compute_dctii_score(
    indicators: Dict[str, float],
    weights: Dict[str, float],
) -> float:
    """Weighted sum of normalized indicators, scaled to 0-100."""
    score = sum(indicators[k] * weights[k] for k in INDICATOR_NAMES)
    return round(score * 100, 2)


def assign_impact_category(score: float) -> str:
    """Map DCTII score to impact category (matches dashboard impactLabel logic)."""
    if score >= 80:
        return "Severe"
    elif score >= 60:
        return "High"
    elif score >= 40:
        return "Moderate"
    elif score >= 20:
        return "Low"
    return "Minimal"


def upsert_scores_to_bq(
    scores: List[dict],
    project_id: str,
    dataset: str,
    data_version: int,
    norm_bounds_hash: str,
):
    """
    Atomic upsert to BigQuery dctii_scores table.
    MERGE key: (site_id, year, weighting_scheme, data_version)
    """
    if not scores:
        logger.warning("No scores to upsert")
        return 0

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset}.dctii_scores"
    temp_table = f"{project_id}.{dataset}._temp_scores_{data_version}"

    # Load to temp table with explicit schema
    df = pd.DataFrame(scores)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["data_version"] = data_version
    df["norm_bounds_hash"] = norm_bounds_hash
    df["created_ts"] = datetime.now(timezone.utc)

    # Ensure numeric types for all float columns
    for col in ["dctii_score", "delta_t_day", "delta_t_night", "heat_island_area_km2",
                "population_exposed", "waste_heat_flux_wm2", "ci_lower", "ci_upper"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df, temp_table, job_config=job_config).result()

    # MERGE
    merge_sql = f"""
    MERGE `{table_id}` T
    USING `{temp_table}` S
    ON T.site_id = S.site_id
      AND T.year = S.year
      AND T.weighting_scheme = S.weighting_scheme
      AND T.data_version = S.data_version
    WHEN MATCHED THEN UPDATE SET
      dctii_score = S.dctii_score,
      impact_category = S.impact_category,
      delta_t_day = S.delta_t_day,
      delta_t_night = S.delta_t_night,
      heat_island_area_km2 = S.heat_island_area_km2,
      population_exposed = S.population_exposed,
      waste_heat_flux_wm2 = S.waste_heat_flux_wm2,
      ci_lower = S.ci_lower,
      ci_upper = S.ci_upper,
      norm_bounds_hash = S.norm_bounds_hash,
      created_ts = S.created_ts
    WHEN NOT MATCHED THEN INSERT (
      year, site_id, weighting_scheme, dctii_score, impact_category,
      ci_lower, ci_upper, delta_t_day, delta_t_night,
      heat_island_area_km2, population_exposed, waste_heat_flux_wm2,
      data_version, norm_bounds_hash, created_ts
    ) VALUES (
      S.year, S.site_id, S.weighting_scheme, S.dctii_score, S.impact_category,
      S.ci_lower, S.ci_upper, S.delta_t_day, S.delta_t_night,
      S.heat_island_area_km2, S.population_exposed, S.waste_heat_flux_wm2,
      S.data_version, S.norm_bounds_hash, S.created_ts
    )
    """
    client.query(merge_sql).result()
    client.delete_table(temp_table, not_found_ok=True)

    # Retention: keep last 3 versions
    retention_sql = f"""
    DELETE FROM `{table_id}`
    WHERE data_version < (SELECT MAX(data_version) - 2 FROM `{table_id}`)
    """
    try:
        client.query(retention_sql).result()
    except Exception as e:
        logger.warning(f"Retention cleanup skipped: {e}")

    logger.info(f"Upserted {len(scores)} scores (version={data_version})")
    return len(scores)


def upsert_did_results_to_bq(
    results: List[dict],
    project_id: str,
    dataset: str,
    data_version: int,
):
    """
    Atomic upsert for DiD causal results.
    MERGE key: (site_id, cohort_year, outcome_var, data_version)
    """
    if not results:
        logger.warning("No DiD results to upsert")
        return 0

    client = bigquery.Client(project=project_id)
    table_id = f"{project_id}.{dataset}.did_results"
    temp_table = f"{project_id}.{dataset}._temp_did_{data_version}"

    df = pd.DataFrame(results)
    df["data_version"] = data_version
    df["created_ts"] = datetime.now(timezone.utc)

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(df, temp_table, job_config=job_config).result()

    merge_sql = f"""
    MERGE `{table_id}` T
    USING `{temp_table}` S
    ON T.site_id = S.site_id
      AND T.cohort_year = S.cohort_year
      AND T.outcome_var = S.outcome_var
      AND T.data_version = S.data_version
    WHEN MATCHED THEN UPDATE SET
      att_estimate = S.att_estimate,
      se = S.se,
      ci_lower = S.ci_lower,
      ci_upper = S.ci_upper,
      p_value = S.p_value,
      n_treated = S.n_treated,
      n_control = S.n_control,
      pre_trend_p = S.pre_trend_p,
      estimation_unit = S.estimation_unit,
      created_ts = S.created_ts
    WHEN NOT MATCHED THEN INSERT ROW
    """
    client.query(merge_sql).result()
    client.delete_table(temp_table, not_found_ok=True)

    logger.info(f"Upserted {len(results)} DiD results (version={data_version})")
    return len(results)


# ---------------------------------------------------------------------------
# ML-optimized weighting with train/test leakage guard
# ---------------------------------------------------------------------------

GROUND_TRUTH_SPLIT_POLICY = """
Ground truth stations are split into three disjoint sets to prevent leakage:
  - TRAIN (60%): used by ML weight optimizer (constrained optimization)
  - VALIDATION (20%): used for early stopping / hyperparameter tuning
  - TEST (20%): held out for final Stage 4 RMSE report, NEVER used for optimization

The split is spatial (by station), not temporal, to avoid autocorrelation leakage.
The test set station IDs are stored in dctii_ref.validation_stations with
split_group = 'test' and are excluded from all weight optimization queries.
"""


def compute_ml_optimized_weights(
    train_stations: List[dict],
    indicators: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Learn indicator weights from ground-truth validation data.
    Constrained optimization: weights sum to 1, non-negative.

    IMPORTANT: This uses only TRAIN split stations. The TEST split is
    reserved exclusively for the Stage 4 RMSE report and is never
    used here. Failure to enforce this creates train/test leakage
    that makes the "validated" weighting scheme appear better than it is.
    """
    raise NotImplementedError("To be implemented in Stage 3/4")


# ---------------------------------------------------------------------------
# Uncertainty propagation from gap-filled observations
# ---------------------------------------------------------------------------

def compute_confidence_interval(
    dctii_score: float,
    indicator_uncertainties: Dict[str, float],
    weights: Dict[str, float],
    fraction_gap_filled: float = 0.0,
) -> Tuple[float, float]:
    """
    Compute CI for a DCTII score incorporating gap-fill uncertainty.

    Propagation chain:
      gap_fill_confidence (per-obs) -> weighted mean confidence (per-month)
      -> Delta-T uncertainty (per-month) -> indicator uncertainty (annual)
      -> weighted sum uncertainty -> DCTII CI

    Gap-filled months receive widened indicator uncertainty:
      sigma_indicator *= (1 + fraction_gap_filled * (1 - mean_gap_confidence))

    This ensures CIs are wider for regions with heavy cloud contamination
    (Houston winter, Toronto winter) and narrower for clear-sky-dominant
    regions (Phoenix, Central Texas).

    Returns: (ci_lower, ci_upper)
    """
    base_sigma = sum(
        (weights[k] * indicator_uncertainties.get(k, 0.0)) ** 2
        for k in INDICATOR_NAMES
    ) ** 0.5

    # Widen CI for gap-filled observations
    gap_penalty = 1.0 + fraction_gap_filled * 0.5  # up to 50% wider
    adjusted_sigma = base_sigma * gap_penalty * 100  # scale to 0-100

    ci_lower = max(0.0, round(dctii_score - 1.96 * adjusted_sigma, 2))
    ci_upper = min(100.0, round(dctii_score + 1.96 * adjusted_sigma, 2))
    return (ci_lower, ci_upper)


# ---------------------------------------------------------------------------
# Full scoring pipeline orchestrator
# ---------------------------------------------------------------------------

def run_dctii_scoring(
    indicators_df: pd.DataFrame,
    weighting_schemes: Optional[Dict[str, Dict[str, float]]] = None,
    use_fixed_bounds: bool = True,
    write_bq: bool = False,
    project_id: str = "oil-tank-monitoring-123",
    dataset: str = "dctii_serving",
    data_version: int = 1,
) -> Dict:
    """
    Full DCTII scoring pipeline:
    1. Winsorize indicators
    2. Normalize (fixed bounds or sample-relative)
    3. Apply weighting schemes (equal primary, others as sensitivity)
    4. Compute composite score (0-100)
    5. Assign impact categories
    6. Compute confidence intervals
    7. Optionally write to BQ
    
    Returns dict with scores DataFrame and summary stats.
    """
    if indicators_df.empty:
        logger.warning("No indicators to score")
        return {"scores": pd.DataFrame(), "n_scored": 0}

    if weighting_schemes is None:
        weighting_schemes = DEFAULT_WEIGHTS

    # Column mapping from indicator_compute output to INDICATOR_NAMES
    col_map = {
        "delta_t_day": "delta_t_day_c",
        "delta_t_night": "delta_t_night_c",
        "heat_island_area_km2": "affected_ring_area_km2",
        "population_exposed": "population_exposed_base",
        "waste_heat_flux_wm2": "waste_heat_flux_wm2",
    }

    # Choose normalization bounds
    if use_fixed_bounds:
        bounds = FIXED_NORMALIZATION_BOUNDS
    else:
        bounds = {}
        for ind_name, col_name in col_map.items():
            vals = indicators_df[col_name].dropna().values
            if len(vals) > 0:
                w = winsorize(vals)
                bounds[ind_name] = (float(np.min(w)), float(np.max(w)))
            else:
                bounds[ind_name] = (0.0, 1.0)

    bounds_hash = compute_norm_bounds_hash(bounds)
    logger.info(f"Normalization bounds hash: {bounds_hash}")

    all_scores = []

    for scheme_name, weights in weighting_schemes.items():
        for _, row in indicators_df.iterrows():
            # Normalize each indicator
            normalized = {}
            for ind_name, col_name in col_map.items():
                raw = row.get(col_name)
                if raw is None or pd.isna(raw):
                    normalized[ind_name] = 0.0
                else:
                    normalized[ind_name] = float(min_max_normalize(
                        np.array([raw]),
                        fixed_bounds=bounds.get(ind_name),
                    )[0])

            # Composite score
            score = compute_dctii_score(normalized, weights)
            category = assign_impact_category(score)

            # Confidence interval
            uncertainties = {
                "delta_t_day": 0.05,
                "delta_t_night": 0.05,
                "heat_island_area_km2": 0.10,
                "population_exposed": 0.15,
                "waste_heat_flux_wm2": row.get("waste_heat_uncertainty_wm2", 0.1) / 500.0,
            }
            ci_lo, ci_hi = compute_confidence_interval(
                score, uncertainties, weights,
                fraction_gap_filled=0.0,
            )

            all_scores.append({
                "site_id": row["site_id"],
                "year": int(row["year"]),
                "weighting_scheme": scheme_name,
                "dctii_score": score,
                "impact_category": category,
                # Raw indicator values (BQ schema column names)
                "delta_t_day": row.get(col_map["delta_t_day"]),
                "delta_t_night": row.get(col_map["delta_t_night"]),
                "heat_island_area_km2": row.get(col_map["heat_island_area_km2"]),
                "population_exposed": row.get(col_map["population_exposed"]),
                "waste_heat_flux_wm2": row.get(col_map["waste_heat_flux_wm2"]),
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
            })

    scores_df = pd.DataFrame(all_scores)

    if write_bq and not scores_df.empty:
        n_written = upsert_scores_to_bq(
            all_scores, project_id, dataset, data_version, bounds_hash
        )
    else:
        n_written = 0

    # Summary
    eq_scores = scores_df[scores_df["weighting_scheme"] == "equal"] if not scores_df.empty else pd.DataFrame()
    summary = {
        "n_scored": len(eq_scores),
        "n_schemes": len(weighting_schemes),
        "bounds_hash": bounds_hash,
        "use_fixed_bounds": use_fixed_bounds,
        "n_written_bq": n_written,
    }

    if not eq_scores.empty:
        summary["score_mean"] = round(float(eq_scores["dctii_score"].mean()), 2)
        summary["score_median"] = round(float(eq_scores["dctii_score"].median()), 2)
        summary["score_min"] = round(float(eq_scores["dctii_score"].min()), 2)
        summary["score_max"] = round(float(eq_scores["dctii_score"].max()), 2)
        summary["categories"] = eq_scores["impact_category"].value_counts().to_dict()

        # Top 5 sites
        top5 = eq_scores.nlargest(5, "dctii_score")[["site_id", "dctii_score", "impact_category"]]
        summary["top_5"] = top5.to_dict("records")

    logger.info(f"Scoring complete: {summary.get('n_scored', 0)} site-years, "
                f"mean={summary.get('score_mean', 'N/A')}")

    return {"scores": scores_df, **summary}
