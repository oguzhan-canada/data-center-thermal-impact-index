"""
validate_retrospective.py — Retrospective Validation of v1 Predictions

Runs the trained v1 model against all 42 existing sites in site_registry,
compares predicted DT values to actual site_indicators labels, and reports:
  1. Per-site predicted vs actual DT night/day
  2. Correlation (Pearson r, Spearman ρ) and MAE
  3. Distribution shift labels — all known sites should be in_distribution or moderate_shift
  4. DCTII score comparison: predicted vs actual (expert scheme)
  5. Summary pass/fail table

Usage:
    python scripts/validate_retrospective.py
"""

import os
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from google.cloud import bigquery

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.predict_train import (
    engineer_features, FEATURE_COLUMNS, apply_bias_offset,
    CLIMATE_ZONE_HEAT_RANK, compute_distribution_shift,
    load_model_artifacts, build_training_matrix,
)
from pipeline.predict_infer import (
    predict_day, compose_dctii_score, derive_day_from_night,
)
from pipeline.dctii_calculator import assign_impact_category

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_retro")

GCP_PROJECT = os.environ.get("GCP_PROJECT", "oil-tank-monitoring-123")

# -- 1. Load training matrix (already has actuals + features) -------------

RETRO_SQL = """
WITH latest_cov AS (
    SELECT site_id,
           ndvi_max AS ndvi_growing_max,
           impervious_fraction, tree_cover_fraction, bare_fraction,
           population_density, elevation_mean_m AS elevation_m, snow_cover_days,
           ROW_NUMBER() OVER (PARTITION BY site_id
                              ORDER BY covariate_date DESC) AS rn
    FROM `{project}.dctii_staging.site_covariates`
    WHERE zone_name = 'footprint'
),
latest_ind AS (
    SELECT site_id, year, delta_t_night, delta_t_day,
           heat_island_area_km2, population_exposed, waste_heat_flux_wm2,
           ROW_NUMBER() OVER (PARTITION BY site_id
                              ORDER BY year DESC) AS rn
    FROM `{project}.dctii_curated.site_indicators`
    WHERE delta_t_night IS NOT NULL
),
latest_score_version AS (
    SELECT MAX(data_version) AS dv
    FROM `{project}.dctii_serving.dctii_scores`
)
SELECT
    r.site_id,
    r.latitude, r.longitude, r.climate_zone,
    r.capacity_mw, r.pue_estimate, r.load_factor, r.cooling_type,
    r.footprint_km2, r.activation_year, r.confidence_tier, r.cluster_id,

    c.ndvi_growing_max, c.impervious_fraction, c.tree_cover_fraction,
    c.bare_fraction, c.population_density, c.elevation_m, c.snow_cover_days,
    CAST(NULL AS FLOAT64) AS era5_solar_mean,
    CAST(NULL AS FLOAT64) AS era5_wind_speed,
    CAST(NULL AS FLOAT64) AS era5_diurnal_range,

    r.capacity_mw * r.load_factor * (r.pue_estimate - 1.0)
        / NULLIF(r.footprint_km2, 0) AS waste_heat_flux_computed,

    i.delta_t_night AS actual_dt_night,
    i.delta_t_day   AS actual_dt_day,
    i.heat_island_area_km2 AS actual_heat_area,
    i.population_exposed   AS actual_pop_exposed,
    i.waste_heat_flux_wm2  AS actual_waste_heat,

    s.dctii_score AS actual_dctii_score

FROM `{project}.dctii_ref.site_registry` r
LEFT JOIN latest_cov c ON c.site_id = r.site_id AND c.rn = 1
LEFT JOIN latest_ind i ON i.site_id = r.site_id AND i.rn = 1
LEFT JOIN latest_score_version lsv ON TRUE
LEFT JOIN `{project}.dctii_serving.dctii_scores` s
    ON s.site_id = r.site_id
    AND s.year = i.year
    AND s.weighting_scheme = 'expert'
    AND s.data_version = lsv.dv
ORDER BY r.site_id
"""


def main():
    print("=" * 80)
    print("DCTII-Predict v1 — Retrospective Validation (all 42 sites)")
    print("=" * 80)

    # -- Load model artifacts from GCS ------------------------------------
    logger.info("Loading model artifacts from GCS...")
    models = load_model_artifacts("latest")
    version = models["eval_report"].get("version", "unknown")
    corrections = models["corrections"]
    distribution = models["distribution"]
    logger.info(f"Model version: {version}")

    # -- Fetch site data from BQ ------------------------------------------
    logger.info("Fetching site data from BigQuery...")
    client = bigquery.Client(project=GCP_PROJECT)
    sql = RETRO_SQL.format(project=GCP_PROJECT)
    df = client.query(sql).to_dataframe()
    logger.info(f"Fetched {len(df)} sites from site_registry")

    if len(df) == 0:
        logger.error("No sites found — check BQ query")
        sys.exit(1)

    # -- Engineer features ------------------------------------------------
    df_feat = engineer_features(df)

    # -- Run predictions for each site ------------------------------------
    results = []
    for idx, row in df_feat.iterrows():
        site_id = row["site_id"]
        climate_zone = row.get("climate_zone", "Cfa")
        climate_rank = int(CLIMATE_ZONE_HEAT_RANK.get(climate_zone, 3))

        X = row[FEATURE_COLUMNS].values.astype(float).reshape(1, -1)

        # Night prediction
        dt_night_raw = float(models["night_median"].predict(X)[0])
        night_offsets = corrections.get("night_bias_offsets", {})
        dt_night = apply_bias_offset(dt_night_raw, climate_rank, night_offsets)
        dt_night_q10 = float(models["night_q10"].predict(X)[0])
        dt_night_q90 = float(models["night_q90"].predict(X)[0])
        night_corr = corrections.get("night_correction", 0.0)

        # Day prediction (stratified routing)
        site_ctx = {
            "is_cluster_site": bool(row.get("is_cluster_site", 0)),
            "region_hint": site_id.split("_")[0] if "_" in site_id else "",
            "climate_heat_rank": climate_rank,
        }
        dt_day, dt_day_lo, dt_day_hi, day_method = predict_day(models, X, site_ctx)

        # Distribution shift
        dist_score, dist_label = compute_distribution_shift(X[0], distribution)

        # Waste heat flux (from engineered features)
        waste_heat = float(row.get("waste_heat_flux", 0.0))

        # DCTII score: use predicted DT + actual auxiliary where available
        heat_area = float(row.get("actual_heat_area", 0.0) or 0.0)
        pop_exposed = float(row.get("actual_pop_exposed", 0.0) or 0.0)
        pred_score, pred_cat = compose_dctii_score(
            dt_day=dt_day, dt_night=dt_night,
            heat_area=heat_area, pop_exposed=pop_exposed,
            waste_heat=waste_heat, scheme="expert",
        )

        results.append({
            "site_id": site_id,
            "climate_zone": climate_zone,
            "climate_rank": climate_rank,
            "capacity_mw": float(row.get("capacity_mw", 0)),
            "cooling_type": row.get("cooling_type", "unknown"),
            # Predicted
            "pred_dt_night": round(dt_night, 4),
            "pred_dt_day": round(dt_day, 4),
            "pred_dt_night_ci": [round(max(0, dt_night_q10 - night_corr), 3),
                                 round(max(0, dt_night_q90 + night_corr), 3)],
            "day_method": day_method,
            "pred_dctii_score": pred_score,
            "pred_impact_cat": pred_cat,
            # Actual
            "actual_dt_night": float(row.get("actual_dt_night") or np.nan),
            "actual_dt_day": float(row.get("actual_dt_day") or np.nan),
            "actual_dctii_score": float(row.get("actual_dctii_score") or np.nan),
            # Shift
            "dist_shift_score": round(dist_score, 2),
            "dist_shift_label": dist_label,
        })

    res_df = pd.DataFrame(results)

    # -- Print per-site results -------------------------------------------
    print("\n" + "-" * 100)
    print(f"{'site_id':<18} {'zone':<5} {'MW':>5} "
          f"{'pred_night':>10} {'act_night':>10} {'D_night':>8} "
          f"{'pred_day':>9} {'act_day':>8} {'D_day':>7} "
          f"{'score_p':>7} {'score_a':>7} {'shift':<16}")
    print("-" * 100)

    for _, r in res_df.iterrows():
        dn = r["pred_dt_night"] - r["actual_dt_night"] if not np.isnan(r["actual_dt_night"]) else np.nan
        dd = r["pred_dt_day"] - r["actual_dt_day"] if not np.isnan(r["actual_dt_day"]) else np.nan
        print(f"{r['site_id']:<18} {r['climate_zone']:<5} {r['capacity_mw']:5.0f} "
              f"{r['pred_dt_night']:10.3f} {r['actual_dt_night']:10.3f} {dn:8.3f} "
              f"{r['pred_dt_day']:9.3f} {r['actual_dt_day']:8.3f} {dd:7.3f} "
              f"{r['pred_dctii_score']:7.1f} {r['actual_dctii_score']:7.1f} {r['dist_shift_label']:<16}")

    # -- Aggregate metrics ------------------------------------------------
    valid_night = res_df.dropna(subset=["actual_dt_night"])
    valid_day = res_df.dropna(subset=["actual_dt_day"])
    valid_score = res_df.dropna(subset=["actual_dctii_score"])

    print("\n" + "=" * 80)
    print("AGGREGATE METRICS")
    print("=" * 80)

    # Night metrics
    if len(valid_night) >= 3:
        mae_night = mean_absolute_error(valid_night["actual_dt_night"], valid_night["pred_dt_night"])
        r2_night = r2_score(valid_night["actual_dt_night"], valid_night["pred_dt_night"])
        pearson_night, _ = stats.pearsonr(valid_night["actual_dt_night"], valid_night["pred_dt_night"])
        spearman_night, sp_p = stats.spearmanr(valid_night["actual_dt_night"], valid_night["pred_dt_night"])
        mbe_night = float(np.mean(valid_night["pred_dt_night"] - valid_night["actual_dt_night"]))

        print(f"\n  DT Night  (n={len(valid_night)}):")
        print(f"    MAE          = {mae_night:.4f} °C")
        print(f"    R²           = {r2_night:.4f}")
        print(f"    Pearson r    = {pearson_night:.4f}")
        print(f"    Spearman ρ   = {spearman_night:.4f} (p={sp_p:.4f})")
        print(f"    MBE          = {mbe_night:+.4f} °C")
    else:
        print("\n  DT Night: insufficient data")

    # Day metrics
    if len(valid_day) >= 3:
        mae_day = mean_absolute_error(valid_day["actual_dt_day"], valid_day["pred_dt_day"])
        r2_day = r2_score(valid_day["actual_dt_day"], valid_day["pred_dt_day"])
        pearson_day, _ = stats.pearsonr(valid_day["actual_dt_day"], valid_day["pred_dt_day"])
        spearman_day, sp_d_p = stats.spearmanr(valid_day["actual_dt_day"], valid_day["pred_dt_day"])
        mbe_day = float(np.mean(valid_day["pred_dt_day"] - valid_day["actual_dt_day"]))

        print(f"\n  DT Day  (n={len(valid_day)}):")
        print(f"    MAE          = {mae_day:.4f} °C")
        print(f"    R²           = {r2_day:.4f}")
        print(f"    Pearson r    = {pearson_day:.4f}")
        print(f"    Spearman ρ   = {spearman_day:.4f} (p={sp_d_p:.4f})")
        print(f"    MBE          = {mbe_day:+.4f} °C")
    else:
        print("\n  DT Day: insufficient data")

    # DCTII score correlation
    if len(valid_score) >= 3:
        score_corr, _ = stats.pearsonr(valid_score["actual_dctii_score"], valid_score["pred_dctii_score"])
        score_mae = mean_absolute_error(valid_score["actual_dctii_score"], valid_score["pred_dctii_score"])
        print(f"\n  DCTII Score  (n={len(valid_score)}):")
        print(f"    Pearson r    = {score_corr:.4f}")
        print(f"    MAE          = {score_mae:.2f} pts")
    else:
        print("\n  DCTII Score: insufficient data")

    # -- Distribution shift summary ---------------------------------------
    print("\n" + "=" * 80)
    print("DISTRIBUTION SHIFT SUMMARY")
    print("=" * 80)
    shift_counts = res_df["dist_shift_label"].value_counts()
    for label, count in shift_counts.items():
        flag = "[OK]" if label in ("in_distribution", "moderate_shift") else "[X]"
        print(f"  {flag}  {label:<20s} : {count} sites")

    bad_shifts = res_df[~res_df["dist_shift_label"].isin(["in_distribution", "moderate_shift"])]
    if len(bad_shifts) > 0:
        print(f"\n  [!] {len(bad_shifts)} sites outside expected distribution:")
        for _, bs in bad_shifts.iterrows():
            print(f"    {bs['site_id']:<18s} shift={bs['dist_shift_score']:.2f} ({bs['dist_shift_label']})")
    else:
        print(f"\n  [OK] All {len(res_df)} sites are in_distribution or moderate_shift")

    # -- Pass / Fail summary ----------------------------------------------
    print("\n" + "=" * 80)
    print("VALIDATION CHECKLIST")
    print("=" * 80)

    checks = []

    if len(valid_night) >= 3:
        checks.append(("Night MAE < 0.50°C", mae_night < 0.50))
        checks.append(("Night Pearson r > 0.70", pearson_night > 0.70))
        checks.append(("Night |MBE| < 0.20°C", abs(mbe_night) < 0.20))

    if len(valid_day) >= 3:
        checks.append(("Day MAE < 0.50°C", mae_day < 0.50))
        checks.append(("Day Pearson r > 0.50", pearson_day > 0.50))

    if len(valid_score) >= 3:
        checks.append(("DCTII score r > 0.60", score_corr > 0.60))

    checks.append(("All sites in/moderate shift", len(bad_shifts) == 0))
    checks.append(("Sites processed = 42", len(res_df) == 42))

    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}]  {label}")

    print("\n" + "=" * 80)
    verdict = "RETROSPECTIVE VALIDATION PASSED" if all_pass else "RETROSPECTIVE VALIDATION: ISSUES FOUND"
    print(f"  {verdict}")
    print("=" * 80)

    # -- Save results to output/ ------------------------------------------
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "retrospective_validation.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    summary = {
        "version": version,
        "n_sites": len(res_df),
        "night": {
            "n": len(valid_night),
            "mae": round(mae_night, 4) if len(valid_night) >= 3 else None,
            "r2": round(r2_night, 4) if len(valid_night) >= 3 else None,
            "pearson_r": round(pearson_night, 4) if len(valid_night) >= 3 else None,
            "spearman_rho": round(spearman_night, 4) if len(valid_night) >= 3 else None,
            "mbe": round(mbe_night, 4) if len(valid_night) >= 3 else None,
        },
        "day": {
            "n": len(valid_day),
            "mae": round(mae_day, 4) if len(valid_day) >= 3 else None,
            "r2": round(r2_day, 4) if len(valid_day) >= 3 else None,
            "pearson_r": round(pearson_day, 4) if len(valid_day) >= 3 else None,
            "spearman_rho": round(spearman_day, 4) if len(valid_day) >= 3 else None,
            "mbe": round(mbe_day, 4) if len(valid_day) >= 3 else None,
        },
        "dctii_score": {
            "n": len(valid_score),
            "pearson_r": round(score_corr, 4) if len(valid_score) >= 3 else None,
            "mae_pts": round(score_mae, 2) if len(valid_score) >= 3 else None,
        },
        "distribution_shift": dict(shift_counts),
        "all_pass": all_pass,
        "checks": {label: passed for label, passed in checks},
    }

    json_path = os.path.join(output_dir, "retrospective_validation.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary saved to: {json_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
