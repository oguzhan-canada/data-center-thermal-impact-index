"""
Populate serving layer: compute indicators, score sites, write to BigQuery.

Usage:
    python -m pipeline.populate_serving [--data-version 1] [--year 2024]

This script:
  1. Reads Delta-T monthly from curated
  2. Computes indicators (5 sub-indicators per site-year)
  3. Writes indicators to dctii_curated.site_indicators
  4. Runs DCTII scoring (normalization + weighting)
  5. Writes scores to dctii_serving.dctii_scores via atomic MERGE
"""

import argparse
import logging
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from google.cloud import bigquery

from pipeline.indicator_compute import compute_all_indicators
from pipeline.dctii_calculator import run_dctii_scoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("dctii.populate")

PROJECT = "oil-tank-monitoring-123"


def load_data(client: bigquery.Client, year: int = None):
    """Load all required data from BigQuery.
    
    The delta_t_monthly table uses wide format (delta_t_day_c + delta_t_night_c
    in same row). We add derived columns (year, method) for downstream compatibility.
    """
    # M-03: Use parameterized queries instead of f-string interpolation
    params = []
    year_filter = ""
    if year:
        year_filter = "WHERE EXTRACT(YEAR FROM year_month) = @year"
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))

    cfg = bigquery.QueryJobConfig(query_parameters=params)
    delta_t_df = client.query(
        f"SELECT * FROM `{PROJECT}.dctii_curated.delta_t_monthly` {year_filter}",
        job_config=cfg,
    ).to_dataframe()

    # Add derived columns for downstream compatibility
    if not delta_t_df.empty:
        delta_t_df["year"] = pd.to_datetime(delta_t_df["year_month"]).dt.year
        # Infer method from site region (NOVA uses ring_difference, others CEM)
        nova_sites = delta_t_df[delta_t_df["site_id"].str.startswith("NOVA")]["site_id"].unique()
        delta_t_df["method"] = np.where(
            delta_t_df["site_id"].isin(nova_sites), "ring_difference", "cem_weighted"
        )
        # Wide format: time_of_day not needed, but add for legacy compat
        delta_t_df["time_of_day"] = "day"  # placeholder; actual day/night in columns

    logger.info(f"Delta-T: {len(delta_t_df)} rows")

    sites_df = client.query(
        f"SELECT * FROM `{PROJECT}.dctii_ref.site_registry`"
    ).to_dataframe()
    logger.info(f"Sites: {len(sites_df)}")

    covariates_df = client.query(
        f"SELECT * FROM `{PROJECT}.dctii_staging.site_covariates`"
    ).to_dataframe()
    logger.info(f"Covariates: {len(covariates_df)}")

    return delta_t_df, sites_df, covariates_df


def compute_heat_area(delta_t_df: pd.DataFrame, sites_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Compute heat island area proxy from Delta-T observations.
    
    For ALL sites (both ring_difference and cem_weighted), uses fraction of
    months where daytime delta_t exceeds threshold as a proxy for thermal
    footprint area. Skips years at or before facility activation.

    NOTE (C-03 deferred): This uses a simplified proxy (fraction_exceeding × 10 km²)
    rather than the zone-area-sum method in anomaly_compute.compute_heat_island_area().
    The two methods produce different values. This proxy is intentionally retained
    until Stage 4 validation confirms which better correlates with ground-truth station
    anomalies. Consolidation is scheduled after multi-year validation sweep.
    Expected max output: 10 km². Zone-area-sum max: ~3,006 km².
    """
    if delta_t_df.empty:
        return pd.DataFrame()

    # Build activation year lookup
    activation_lookup = dict(zip(sites_df["site_id"], sites_df["activation_year"]))

    records = []
    for (site_id, year), grp in delta_t_df.groupby(["site_id", "year"]):
        # Skip years at or before facility activation
        act_year = activation_lookup.get(site_id)
        if act_year and year <= int(act_year):
            continue
        # Use daytime delta_t (wide format: delta_t_day_c column)
        valid_day = grp[grp["delta_t_day_c"].notna()]
        if valid_day.empty:
            fraction_day = 0.0
        else:
            affected_day = valid_day[valid_day["delta_t_day_c"] > threshold]
            fraction_day = len(affected_day) / len(valid_day)

        # Also check nighttime
        valid_night = grp[grp["delta_t_night_c"].notna()]
        if valid_night.empty:
            fraction_night = 0.0
        else:
            affected_night = valid_night[valid_night["delta_t_night_c"] > threshold]
            fraction_night = len(affected_night) / len(valid_night)

        # Use max of day/night fraction (thermal impact is detected in either)
        fraction_affected = max(fraction_day, fraction_night)
        area_est = fraction_affected * 10.0

        records.append({
            "site_id": site_id,
            "year": year,
            "time_of_day": "day",
            "affected_ring_area_km2": round(area_est, 2),
        })

    return pd.DataFrame(records)


def write_indicators_to_bq(
    indicators_df: pd.DataFrame,
    client: bigquery.Client,
):
    """Write indicators to dctii_curated.site_indicators via MERGE."""
    if indicators_df.empty:
        logger.warning("No indicators to write")
        return 0

    table_id = f"{PROJECT}.dctii_curated.site_indicators"
    # M-09: Per-run temp table name to prevent race conditions
    run_id = uuid.uuid4().hex[:8]
    temp_table = f"{PROJECT}.dctii_curated._temp_indicators_{run_id}"

    # Map column names from compute output to BQ schema
    bq_df = pd.DataFrame({
        "year": indicators_df["year"].astype(int),
        "site_id": indicators_df["site_id"],
        "delta_t_day": indicators_df["delta_t_day_c"],
        "delta_t_night": indicators_df["delta_t_night_c"],
        "heat_island_area_km2": indicators_df["affected_ring_area_km2"],
        "population_exposed": indicators_df["population_exposed_base"],
        "waste_heat_flux_wm2": indicators_df["waste_heat_flux_wm2"],
        "min_monthly_reliability": indicators_df.get("estimation_method", "unknown"),
        "fraction_reliable_months": indicators_df["n_months_reliable"].astype(float) / 12.0
            if "n_months_reliable" in indicators_df.columns else 0.0,
        "created_ts": datetime.now(timezone.utc),
    })

    # Normalized values (to be filled by scoring, but store nulls for now)
    for col in ["delta_t_day_norm", "delta_t_night_norm", "heat_island_area_norm",
                "population_exposed_norm", "waste_heat_flux_norm"]:
        bq_df[col] = None

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(bq_df, temp_table, job_config=job_config).result()

    try:
        merge_sql = f"""
        MERGE `{table_id}` T
        USING `{temp_table}` S
        ON T.site_id = S.site_id AND T.year = S.year
        WHEN MATCHED THEN UPDATE SET
          delta_t_day = S.delta_t_day,
          delta_t_night = S.delta_t_night,
          heat_island_area_km2 = S.heat_island_area_km2,
          population_exposed = S.population_exposed,
          waste_heat_flux_wm2 = S.waste_heat_flux_wm2,
          min_monthly_reliability = S.min_monthly_reliability,
          fraction_reliable_months = S.fraction_reliable_months,
          created_ts = S.created_ts
        WHEN NOT MATCHED THEN INSERT (
          year, site_id, delta_t_day, delta_t_night, heat_island_area_km2,
          population_exposed, waste_heat_flux_wm2, min_monthly_reliability,
          fraction_reliable_months, created_ts
        ) VALUES (
          S.year, S.site_id, S.delta_t_day, S.delta_t_night, S.heat_island_area_km2,
          S.population_exposed, S.waste_heat_flux_wm2, S.min_monthly_reliability,
          S.fraction_reliable_months, S.created_ts
        )
        """
        client.query(merge_sql).result()
    finally:
        client.delete_table(temp_table, not_found_ok=True)
    logger.info(f"Wrote {len(bq_df)} indicator rows to site_indicators")
    return len(bq_df)


def main():
    parser = argparse.ArgumentParser(description="Populate DCTII serving layer")
    parser.add_argument("--data-version", type=int, default=1)
    parser.add_argument("--year", type=int, default=None, help="Filter to specific year")
    parser.add_argument("--force", action="store_true",
                        help="Skip idempotency check and re-run (L-06)")
    args = parser.parse_args()

    client = bigquery.Client(project=PROJECT)

    # R-01: Idempotency guard via PipelineRunTracker
    from pipeline.pipeline_run import PipelineRunTracker
    tracker = PipelineRunTracker(
        client, stage="populate_serving", year=args.year,
        triggered_by=os.getenv("TRIGGERED_BY", "manual"),
    )
    if not tracker.start(force=args.force):
        logger.info("Already successful — skipping (use --force to rerun)")
        return

    try:
        # 1. Load data
        logger.info("Loading data from BigQuery...")
        delta_t_df, sites_df, covariates_df = load_data(client, args.year)

        if delta_t_df.empty:
            raise RuntimeError("No Delta-T data found — run anomaly_compute first")

        # 2. Compute heat area proxy
        heat_area_df = compute_heat_area(delta_t_df, sites_df)
        logger.info(f"Heat area records: {len(heat_area_df)}")

        # 3. Compute indicators
        indicators_df = compute_all_indicators(delta_t_df, heat_area_df, sites_df, covariates_df)
        logger.info(f"Indicators: {len(indicators_df)} site-years")

        # R-05: Guard against empty indicators
        if indicators_df.empty:
            raise RuntimeError(
                "compute_all_indicators returned empty DataFrame — "
                "check delta_t_df column names and covariates availability"
            )

        # 4. Write indicators to curated
        n_ind = write_indicators_to_bq(indicators_df, client)

        # 5. Score and write to serving
        result = run_dctii_scoring(
            indicators_df,
            write_bq=True,
            project_id=PROJECT,
            dataset="dctii_serving",
            data_version=args.data_version,
        )
        logger.info(f"Scoring complete: {result.get('n_scored', 0)} site-years scored")
        logger.info(f"  Mean score: {result.get('score_mean', 'N/A')}")
        logger.info(f"  Score range: {result.get('score_min', 'N/A')} - {result.get('score_max', 'N/A')}")
        logger.info(f"  Categories: {result.get('categories', {})}")
        logger.info(f"  BQ writes: {n_ind} indicators + {result.get('n_written_bq', 0)} scores")

        # 6. Verify
        for table in ["dctii_curated.site_indicators", "dctii_serving.dctii_scores"]:
            count = list(client.query(f"SELECT COUNT(*) c FROM `{PROJECT}.{table}`").result())[0].c
            logger.info(f"  {table}: {count} rows")

        tracker.complete(rows_written=n_ind + result.get("n_written_bq", 0))

    except Exception as e:
        tracker.fail(str(e))
        raise


if __name__ == "__main__":
    main()
