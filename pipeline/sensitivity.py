"""
Stage 4.1: Sensitivity Analysis Framework

Two-layer design (per rubber-duck critique):
  Layer 1 — Feature-generation sensitivity:
    Changes that affect indicator computation (threshold, radius, overpass, season, country).
    These re-derive indicators from Delta-T data.
  Layer 2 — Score-composition sensitivity:
    Changes that only affect scoring (weights, normalization, aggregation).
    These re-score from existing indicators.

Output: Variation manifest with Spearman ρ, median/max rank shift, score changes.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
from scipy.stats import spearmanr

logger = logging.getLogger("dctii.sensitivity")


# ---------------------------------------------------------------------------
# Variation configurations
# ---------------------------------------------------------------------------

SCORING_VARIATIONS = {
    # Weight perturbations: each indicator ±25%, renormalized
    "weight_dt_day_up": {"delta_t_day": 0.25, "delta_t_night": 0.1875,
                         "heat_island_area_km2": 0.1875, "population_exposed": 0.1875,
                         "waste_heat_flux_wm2": 0.1875},
    "weight_dt_day_down": {"delta_t_day": 0.15, "delta_t_night": 0.2125,
                           "heat_island_area_km2": 0.2125, "population_exposed": 0.2125,
                           "waste_heat_flux_wm2": 0.2125},
    "weight_dt_night_up": {"delta_t_day": 0.1875, "delta_t_night": 0.25,
                           "heat_island_area_km2": 0.1875, "population_exposed": 0.1875,
                           "waste_heat_flux_wm2": 0.1875},
    "weight_waste_heat_up": {"delta_t_day": 0.1875, "delta_t_night": 0.1875,
                             "heat_island_area_km2": 0.1875, "population_exposed": 0.1875,
                             "waste_heat_flux_wm2": 0.25},
    "weight_pop_up": {"delta_t_day": 0.1875, "delta_t_night": 0.1875,
                      "heat_island_area_km2": 0.1875, "population_exposed": 0.25,
                      "waste_heat_flux_wm2": 0.1875},

    # Normalization method variations
    "norm_zscore": "zscore",
    "norm_rank": "rank",
    "norm_minmax_sample": "minmax_sample",  # sample-relative instead of fixed

    # Aggregation method variations
    "agg_geometric": "geometric",
    "agg_maximum": "maximum",
}

FEATURE_VARIATIONS = {
    # Threshold variations for heat island area
    "threshold_0.2": {"threshold_c": 0.2},
    "threshold_0.5": {"threshold_c": 0.5},  # baseline
    "threshold_1.0": {"threshold_c": 1.0},

    # Overpass time
    "overpass_day_only": {"time_filter": "day"},
    "overpass_combined": {"time_filter": "all"},  # baseline

    # Season (Canadian relevance)
    "season_summer": {"season_filter": "JJA"},
    "season_winter": {"season_filter": "DJF"},
    "season_annual": {"season_filter": "annual"},  # baseline

    # Country subset
    "country_us": {"country_filter": "US"},
    "country_ca": {"country_filter": "CA"},
    "country_combined": {"country_filter": "all"},  # baseline

    # Method filter
    "method_cem_only": {"method_filter": "cem_weighted"},
    "method_ring_only": {"method_filter": "ring_difference"},
}

# Fixed normalization bounds (baseline)
FIXED_BOUNDS = {
    "delta_t_day": (0.0, 8.0),
    "delta_t_night": (0.0, 8.0),
    "heat_island_area_km2": (0.0, 50.0),
    "population_exposed": (0.0, 100000.0),
    "waste_heat_flux_wm2": (0.0, 500.0),
}

INDICATOR_NAMES = [
    "delta_t_day", "delta_t_night", "heat_island_area_km2",
    "population_exposed", "waste_heat_flux_wm2",
]

# Column mapping from indicator_compute output
COL_MAP = {
    "delta_t_day": "delta_t_day_c",
    "delta_t_night": "delta_t_night_c",
    "heat_island_area_km2": "affected_ring_area_km2",
    "population_exposed": "population_exposed_base",
    "waste_heat_flux_wm2": "waste_heat_flux_wm2",
}


# ---------------------------------------------------------------------------
# Feature-generation layer (Layer 1)
# ---------------------------------------------------------------------------

def apply_feature_variation(
    delta_t_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Re-derive indicators from Delta-T with varied feature config.
    Returns modified indicators DataFrame.
    """
    df = delta_t_df.copy()

    # Time filter
    time_filter = config.get("time_filter")
    if time_filter and time_filter != "all":
        df = df[df["time_of_day"] == time_filter]

    # Season filter
    season_filter = config.get("season_filter")
    if season_filter and season_filter != "annual":
        month_map = {
            "JJA": [6, 7, 8],
            "DJF": [12, 1, 2],
            "MAM": [3, 4, 5],
            "SON": [9, 10, 11],
        }
        months = month_map.get(season_filter, list(range(1, 13)))
        if "year_month" in df.columns:
            df["_month"] = pd.to_datetime(df["year_month"]).dt.month
        elif "month" in df.columns:
            df["_month"] = df["month"]
        else:
            df["_month"] = 1
        df = df[df["_month"].isin(months)]
        df = df.drop(columns=["_month"], errors="ignore")

    # Country filter
    country_filter = config.get("country_filter")
    if country_filter and country_filter != "all":
        ca_regions = {"TOR", "MTL"}
        if country_filter == "CA":
            df = df[df["region_code"].isin(ca_regions)]
        elif country_filter == "US":
            df = df[~df["region_code"].isin(ca_regions)]

    # Method filter
    method_filter = config.get("method_filter")
    if method_filter:
        df = df[df["method"] == method_filter]

    if df.empty:
        return pd.DataFrame()

    # Re-aggregate to site-year indicators
    results = []
    for (site_id, year), grp in df.groupby(["site_id", "year"]):
        site_info = sites_df[sites_df["site_id"] == site_id]
        if site_info.empty:
            continue

        # Find matching indicator row for waste heat (doesn't change)
        ind_row = indicators_df[
            (indicators_df["site_id"] == site_id) &
            (indicators_df["year"] == year)
        ]

        day = grp[grp["time_of_day"] == "day"] if "time_of_day" in grp.columns else grp
        night = grp[grp["time_of_day"] == "night"] if "time_of_day" in grp.columns else pd.DataFrame()

        dt_day = float(day["delta_t_day_c"].mean()) if not day.empty and "delta_t_day_c" in day.columns else None
        dt_night = float(night["delta_t_day_c"].mean()) if not night.empty and "delta_t_day_c" in night.columns else None

        # Heat island area with threshold
        threshold = config.get("threshold_c", 0.5)
        if not day.empty and "delta_t_day_c" in day.columns:
            affected = day[day["delta_t_day_c"] > threshold]
            area = len(affected) * 0.5  # simplified proxy
            area = min(area, 50.0)
        else:
            area = 0.0

        results.append({
            "site_id": site_id,
            "region_code": grp["region_code"].iloc[0] if "region_code" in grp.columns else "",
            "year": year,
            "delta_t_day_c": dt_day,
            "delta_t_night_c": dt_night,
            "affected_ring_area_km2": area,
            "population_exposed_base": float(ind_row["population_exposed_base"].iloc[0]) if not ind_row.empty else 0.0,
            "waste_heat_flux_wm2": float(ind_row["waste_heat_flux_wm2"].iloc[0]) if not ind_row.empty else 0.0,
            "waste_heat_uncertainty_wm2": float(ind_row["waste_heat_uncertainty_wm2"].iloc[0]) if not ind_row.empty else 0.0,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Score-composition layer (Layer 2)
# ---------------------------------------------------------------------------

def normalize_indicators(
    indicators_df: pd.DataFrame,
    method: str = "minmax_fixed",
    bounds: Optional[Dict] = None,
) -> pd.DataFrame:
    """Apply normalization to indicators. Returns DataFrame with _norm columns."""
    df = indicators_df.copy()

    for ind_name, col_name in COL_MAP.items():
        vals = df[col_name].fillna(0).values

        if method == "minmax_fixed":
            lo, hi = (bounds or FIXED_BOUNDS).get(ind_name, (0, 1))
            if hi - lo < 1e-9:
                normed = np.zeros_like(vals)
            else:
                normed = np.clip((vals - lo) / (hi - lo), 0, 1)

        elif method == "minmax_sample":
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            if hi - lo < 1e-9:
                normed = np.zeros_like(vals)
            else:
                normed = np.clip((vals - lo) / (hi - lo), 0, 1)

        elif method == "zscore":
            mu, sigma = np.nanmean(vals), np.nanstd(vals)
            if sigma < 1e-9:
                normed = np.zeros_like(vals)
            else:
                normed = (vals - mu) / sigma
                # Map to [0,1] via CDF approximation for compositing
                from scipy.stats import norm as norm_dist
                normed = norm_dist.cdf(normed)

        elif method == "rank":
            from scipy.stats import rankdata
            normed = (rankdata(vals) - 1) / max(len(vals) - 1, 1)

        else:
            normed = vals

        df[f"{ind_name}_norm"] = normed

    return df


def compute_composite(
    normed_df: pd.DataFrame,
    weights: Dict[str, float],
    aggregation: str = "weighted_sum",
) -> pd.Series:
    """Compute composite score from normalized indicators."""
    norm_cols = {k: f"{k}_norm" for k in INDICATOR_NAMES}

    if aggregation == "weighted_sum":
        score = sum(
            normed_df[norm_cols[k]].fillna(0) * weights[k]
            for k in INDICATOR_NAMES
        ) * 100

    elif aggregation == "geometric":
        # Geometric mean on positive values with epsilon floor
        eps = 0.001
        log_sum = sum(
            np.log(normed_df[norm_cols[k]].fillna(0).clip(lower=eps)) * weights[k]
            for k in INDICATOR_NAMES
        )
        score = np.exp(log_sum) * 100

    elif aggregation == "maximum":
        # Max of weighted normalized indicators
        weighted = pd.DataFrame({
            k: normed_df[norm_cols[k]].fillna(0) * weights[k]
            for k in INDICATOR_NAMES
        })
        score = weighted.max(axis=1) * 100 / max(weights.values())

    else:
        score = pd.Series(0, index=normed_df.index)

    return score.clip(0, 100).round(2)


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def compute_comparison_metrics(
    baseline_scores: pd.Series,
    variant_scores: pd.Series,
    baseline_ranks: pd.Series,
    variant_ranks: pd.Series,
) -> dict:
    """Compute comparison metrics between baseline and variant."""
    # Align on index
    common = baseline_scores.index.intersection(variant_scores.index)
    if len(common) < 3:
        return {"n_common": len(common), "spearman_rho": None}

    b_scores = baseline_scores.loc[common]
    v_scores = variant_scores.loc[common]
    b_ranks = baseline_ranks.loc[common]
    v_ranks = variant_ranks.loc[common]

    rho, p_val = spearmanr(b_scores, v_scores)
    rank_shifts = (v_ranks - b_ranks).abs()

    return {
        "n_common": len(common),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p_val), 4),
        "median_rank_shift": float(rank_shifts.median()),
        "max_rank_shift": float(rank_shifts.max()),
        "mean_score_change": round(float((v_scores - b_scores).mean()), 3),
        "median_abs_score_change": round(float((v_scores - b_scores).abs().median()), 3),
        "max_abs_score_change": round(float((v_scores - b_scores).abs().max()), 3),
        # Top/bottom quintile retention
        "top_quintile_retained": _quintile_retention(b_ranks, v_ranks, "top"),
        "bottom_quintile_retained": _quintile_retention(b_ranks, v_ranks, "bottom"),
    }


def _quintile_retention(base_ranks, var_ranks, which="top"):
    """Fraction of top/bottom quintile that remains in same quintile."""
    n = len(base_ranks)
    q_size = max(n // 5, 1)
    if which == "top":
        base_set = set(base_ranks.nsmallest(q_size).index)
        var_set = set(var_ranks.nsmallest(q_size).index)
    else:
        base_set = set(base_ranks.nlargest(q_size).index)
        var_set = set(var_ranks.nlargest(q_size).index)
    return round(len(base_set & var_set) / max(len(base_set), 1), 3)


# ---------------------------------------------------------------------------
# Main sensitivity sweep
# ---------------------------------------------------------------------------

def run_sensitivity_sweep(
    delta_t_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run full sensitivity sweep across all variations.
    
    Returns DataFrame with one row per variation:
      variation_name, layer, spearman_rho, median_rank_shift, max_rank_shift,
      mean_score_change, top_quintile_retained, etc.
    """
    if indicators_df.empty:
        logger.warning("No indicators for sensitivity sweep")
        return pd.DataFrame()

    # Equal weights baseline
    base_weights = {k: 0.2 for k in INDICATOR_NAMES}

    # Compute baseline scores
    base_normed = normalize_indicators(indicators_df, method="minmax_fixed")
    base_scores = compute_composite(base_normed, base_weights, "weighted_sum")
    base_scores.index = indicators_df["site_id"]
    base_ranks = base_scores.rank(ascending=False)

    results = []

    # --- Layer 2: Scoring variations (fast, no feature re-computation) ---
    for var_name, var_config in SCORING_VARIATIONS.items():
        logger.info(f"Scoring variation: {var_name}")

        if isinstance(var_config, dict):
            # Weight variation
            weights = var_config
            normed = base_normed
            agg = "weighted_sum"
        elif var_config in ("zscore", "rank", "minmax_sample"):
            # Normalization variation
            weights = base_weights
            normed = normalize_indicators(indicators_df, method=var_config)
            agg = "weighted_sum"
        elif var_config == "geometric":
            weights = base_weights
            normed = base_normed
            agg = "geometric"
        elif var_config == "maximum":
            weights = base_weights
            normed = base_normed
            agg = "maximum"
        else:
            continue

        var_scores = compute_composite(normed, weights, agg)
        var_scores.index = indicators_df["site_id"]
        var_ranks = var_scores.rank(ascending=False)

        metrics = compute_comparison_metrics(base_scores, var_scores, base_ranks, var_ranks)
        metrics["variation"] = var_name
        metrics["layer"] = "scoring"
        results.append(metrics)

    # --- Layer 1: Feature variations (re-derive indicators) ---
    for var_name, var_config in FEATURE_VARIATIONS.items():
        logger.info(f"Feature variation: {var_name}")

        var_indicators = apply_feature_variation(
            delta_t_df, indicators_df, sites_df, var_config
        )

        if var_indicators.empty or len(var_indicators) < 3:
            logger.warning(f"  Skipped {var_name}: too few sites ({len(var_indicators)})")
            continue

        var_normed = normalize_indicators(var_indicators, method="minmax_fixed")
        var_scores = compute_composite(var_normed, base_weights, "weighted_sum")
        var_scores.index = var_indicators["site_id"]
        var_ranks = var_scores.rank(ascending=False)

        metrics = compute_comparison_metrics(base_scores, var_scores, base_ranks, var_ranks)
        metrics["variation"] = var_name
        metrics["layer"] = "feature"
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Summary
    if not results_df.empty:
        passing = results_df[results_df["spearman_rho"].notna() & (results_df["spearman_rho"] >= 0.7)]
        logger.info(f"Sensitivity sweep: {len(results_df)} variations, "
                     f"{len(passing)}/{len(results_df)} pass ρ≥0.7 threshold")

    return results_df
