"""
Stage 4.2-4.3: Convergent Validation & Cloud Bias Diagnostics

Ground-truth validation against NOAA ISD weather stations via BigQuery
public dataset. Framed as CONVERGENT VALIDATION (not strict ground truth)
because:
  - Satellite LST measures surface temperature; stations measure 2m air temp
  - Station siting biases (airports, pavement) affect representativeness
  - Temporal alignment (overpass vs hourly) introduces uncertainty

Cloud bias diagnostics:
  - Cloud-free fraction by site/sensor/season/day-night
  - Delta-T ~ cloud fraction regression with region/month FE
  - Differential missingness between treated and reference areas
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from google.cloud import bigquery
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger("dctii.validation")

GCP_PROJECT = "oil-tank-monitoring-123"

# Site coordinates for station matching (from site_registry)
REGION_COUNTRIES = {
    "PHX": "US", "HOU": "US", "NOVA": "US", "CTX": "US",
    "TOR": "CA", "MTL": "CA",
}


# ---------------------------------------------------------------------------
# 4.2 Station-based convergent validation
# ---------------------------------------------------------------------------

def find_nearby_stations(
    sites_df: pd.DataFrame,
    max_distance_km: float = 10.0,
    min_years: int = 3,
) -> pd.DataFrame:
    """
    Find weather stations near DC sites.
    Uses curated station list for our 6 study regions.
    
    For production, supplement with NOAA ISD API or GHCNd.
    """
    # Curated stations near our study regions (from NOAA station finder)
    # Each has: station_id, name, lat, lon, region, is_airport
    KNOWN_STATIONS = [
        # Phoenix
        {"usaf": "722780", "wban": "23183", "station_name": "PHOENIX SKY HARBOR INTL", "station_lat": 33.434, "station_lon": -111.994, "region": "PHX", "is_airport": True},
        {"usaf": "722784", "wban": "03192", "station_name": "PHOENIX DEER VALLEY", "station_lat": 33.688, "station_lon": -112.083, "region": "PHX", "is_airport": True},
        {"usaf": "722785", "wban": "23104", "station_name": "MESA FALCON FIELD", "station_lat": 33.458, "station_lon": -111.728, "region": "PHX", "is_airport": True},
        {"usaf": "722789", "wban": "99999", "station_name": "CHANDLER MUNICIPAL", "station_lat": 33.269, "station_lon": -111.811, "region": "PHX", "is_airport": True},
        # Houston
        {"usaf": "722430", "wban": "12960", "station_name": "HOUSTON HOBBY", "station_lat": 29.645, "station_lon": -95.279, "region": "HOU", "is_airport": True},
        {"usaf": "722435", "wban": "12918", "station_name": "HOUSTON IAH", "station_lat": 29.980, "station_lon": -95.360, "region": "HOU", "is_airport": True},
        {"usaf": "722436", "wban": "99999", "station_name": "HOUSTON ELLINGTON", "station_lat": 29.607, "station_lon": -95.159, "region": "HOU", "is_airport": True},
        # Northern Virginia
        {"usaf": "724030", "wban": "93738", "station_name": "WASHINGTON DULLES INTL", "station_lat": 38.935, "station_lon": -77.447, "region": "NOVA", "is_airport": True},
        {"usaf": "724036", "wban": "93741", "station_name": "STERLING", "station_lat": 38.976, "station_lon": -77.487, "region": "NOVA", "is_airport": False},
        {"usaf": "724050", "wban": "13743", "station_name": "WASHINGTON REAGAN", "station_lat": 38.851, "station_lon": -77.034, "region": "NOVA", "is_airport": True},
        # Central Texas
        {"usaf": "722540", "wban": "13958", "station_name": "AUSTIN BERGSTROM", "station_lat": 30.194, "station_lon": -97.670, "region": "CTX", "is_airport": True},
        {"usaf": "722544", "wban": "99999", "station_name": "SAN MARCOS MUNI", "station_lat": 29.894, "station_lon": -97.863, "region": "CTX", "is_airport": True},
        {"usaf": "722560", "wban": "13959", "station_name": "SAN ANTONIO INTL", "station_lat": 29.534, "station_lon": -98.470, "region": "CTX", "is_airport": True},
        # Toronto
        {"usaf": "716240", "wban": "99999", "station_name": "TORONTO PEARSON INTL", "station_lat": 43.677, "station_lon": -79.631, "region": "TOR", "is_airport": True},
        {"usaf": "716243", "wban": "99999", "station_name": "TORONTO CITY CENTRE", "station_lat": 43.627, "station_lon": -79.396, "region": "TOR", "is_airport": True},
        {"usaf": "716245", "wban": "99999", "station_name": "TORONTO BUTTONVILLE", "station_lat": 43.862, "station_lon": -79.370, "region": "TOR", "is_airport": True},
        # Montreal
        {"usaf": "716270", "wban": "99999", "station_name": "MONTREAL TRUDEAU INTL", "station_lat": 45.470, "station_lon": -73.741, "region": "MTL", "is_airport": True},
        {"usaf": "716278", "wban": "99999", "station_name": "MONTREAL ST-HUBERT", "station_lat": 45.517, "station_lon": -73.417, "region": "MTL", "is_airport": True},
        {"usaf": "716272", "wban": "99999", "station_name": "MONTREAL MIRABEL", "station_lat": 45.680, "station_lon": -74.039, "region": "MTL", "is_airport": True},
    ]

    station_df = pd.DataFrame(KNOWN_STATIONS)

    # Compute distances to each DC site
    results = []
    for _, site in sites_df.iterrows():
        site_lat = float(site["latitude"])
        site_lon = float(site["longitude"])
        region = site["region_code"]

        # Only match stations in same region
        region_stations = station_df[station_df["region"] == region]

        for _, stn in region_stations.iterrows():
            # Haversine distance
            dlat = np.radians(stn["station_lat"] - site_lat)
            dlon = np.radians(stn["station_lon"] - site_lon)
            a = (np.sin(dlat / 2) ** 2 +
                 np.cos(np.radians(site_lat)) * np.cos(np.radians(stn["station_lat"])) *
                 np.sin(dlon / 2) ** 2)
            dist_km = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            if dist_km <= max_distance_km:
                results.append({
                    "site_id": site["site_id"],
                    "site_lat": site_lat,
                    "site_lon": site_lon,
                    "region": region,
                    "usaf": stn["usaf"],
                    "wban": stn["wban"],
                    "station_name": stn["station_name"],
                    "station_lat": stn["station_lat"],
                    "station_lon": stn["station_lon"],
                    "distance_km": round(dist_km, 2),
                    "is_airport": stn["is_airport"],
                })

    df = pd.DataFrame(results)
    if not df.empty:
        df["rank_score"] = df["distance_km"] + df["is_airport"].astype(int) * 5.0
        df = df.sort_values(["site_id", "rank_score"])

    logger.info(f"Found {len(df)} station-site pairs within {max_distance_km} km")
    return df


def fetch_station_temps(
    usaf_wban_pairs: List[Tuple[str, str]],
    year: int = 2024,
) -> pd.DataFrame:
    """
    Fetch daily temperatures from NOAA GSOD via BigQuery public data.
    Falls back to synthetic validation data if BQ public access unavailable.
    """
    client = bigquery.Client(project=GCP_PROJECT)

    station_filters = " OR ".join(
        f"(stn = '{usaf}' AND wban = '{wban}')"
        for usaf, wban in usaf_wban_pairs
    )

    query = f"""
    SELECT
        stn AS usaf, wban,
        PARSE_DATE('%Y%m%d', CONCAT(
            CAST(year AS STRING),
            LPAD(CAST(mo AS STRING), 2, '0'),
            LPAD(CAST(da AS STRING), 2, '0')
        )) AS obs_date,
        CAST(year AS INT64) AS year,
        CAST(mo AS INT64) AS month,
        CAST(da AS INT64) AS day,
        (temp - 32) * 5.0/9.0 AS temp_c,
        (max - 32) * 5.0/9.0 AS max_c,
        (min - 32) * 5.0/9.0 AS min_c,
    FROM `bigquery-public-data.noaa_gsod.gsod{year}`
    WHERE ({station_filters})
      AND temp != 9999.9
    ORDER BY usaf, wban, year, mo, da
    """

    try:
        df = client.query(query).to_dataframe()
        logger.info(f"Fetched {len(df)} daily station records for {year}")
        return df
    except Exception as e:
        logger.warning(f"BQ public GSOD access failed ({e}). Using summary validation.")
        # Return empty — validation will report insufficient data
        return pd.DataFrame()


def compute_station_anomalies(
    station_temps: pd.DataFrame,
    near_stations: pd.DataFrame,  # stations near DCs
    baseline_stations: pd.DataFrame,  # stations far from DCs (>10km)
) -> pd.DataFrame:
    """
    Compute monthly temperature anomalies:
      anomaly = T_near_dc - T_baseline (monthly mean)
    
    This is the air-temperature analog of satellite Delta-T.
    """
    if station_temps.empty:
        return pd.DataFrame()

    # Monthly mean temps per station
    monthly = station_temps.groupby(["usaf", "wban", "month"]).agg(
        temp_c=("temp_c", "mean"),
        max_c=("max_c", "mean"),
        min_c=("min_c", "mean"),
        n_days=("temp_c", "count"),
    ).reset_index()

    # Split into near-DC and baseline
    near_ids = set(zip(near_stations["usaf"], near_stations["wban"]))
    baseline_ids = set(zip(baseline_stations["usaf"], baseline_stations["wban"]))

    near_monthly = monthly[
        monthly.apply(lambda r: (r["usaf"], r["wban"]) in near_ids, axis=1)
    ]
    base_monthly = monthly[
        monthly.apply(lambda r: (r["usaf"], r["wban"]) in baseline_ids, axis=1)
    ]

    # Regional baseline: mean across all baseline stations per month
    regional_baseline = base_monthly.groupby("month").agg(
        baseline_temp=("temp_c", "mean"),
        baseline_max=("max_c", "mean"),
        baseline_min=("min_c", "mean"),
    ).reset_index()

    # Merge and compute anomaly
    anomalies = near_monthly.merge(regional_baseline, on="month", how="left")
    anomalies["temp_anomaly_c"] = anomalies["temp_c"] - anomalies["baseline_temp"]
    anomalies["max_anomaly_c"] = anomalies["max_c"] - anomalies["baseline_max"]
    anomalies["min_anomaly_c"] = anomalies["min_c"] - anomalies["baseline_min"]

    return anomalies


def validate_against_stations(
    satellite_delta_t: pd.DataFrame,
    station_anomalies: pd.DataFrame,
    station_site_map: pd.DataFrame,
) -> dict:
    """
    Compare satellite Delta-T with station air-temp anomalies.
    Returns validation metrics: RMSE, MAE, Pearson r, bias.
    """
    if satellite_delta_t.empty or station_anomalies.empty:
        return {"status": "insufficient_data"}

    # Merge station anomalies with site mapping
    merged = station_anomalies.merge(
        station_site_map[["usaf", "wban", "site_id"]],
        on=["usaf", "wban"],
        how="inner"
    )

    # Merge with satellite monthly Delta-T
    sat_monthly = satellite_delta_t.groupby(["site_id", "year_month"]).agg(
        sat_delta_t=("delta_t_day_c", "mean"),
    ).reset_index()

    # Convert year_month to month for matching
    sat_monthly["month"] = pd.to_datetime(sat_monthly["year_month"]).dt.month

    comparison = merged.merge(
        sat_monthly[["site_id", "month", "sat_delta_t"]],
        on=["site_id", "month"],
        how="inner"
    )

    if comparison.empty or len(comparison) < 5:
        return {"status": "insufficient_overlap", "n_pairs": len(comparison)}

    sat = comparison["sat_delta_t"].values
    station = comparison["temp_anomaly_c"].values

    # Metrics
    rmse = float(np.sqrt(np.mean((sat - station) ** 2)))
    mae = float(np.mean(np.abs(sat - station)))
    bias = float(np.mean(sat - station))
    r, p_val = pearsonr(sat, station)

    return {
        "status": "complete",
        "n_pairs": len(comparison),
        "n_sites": comparison["site_id"].nunique(),
        "rmse_c": round(rmse, 3),
        "mae_c": round(mae, 3),
        "bias_c": round(bias, 3),
        "pearson_r": round(float(r), 4),
        "pearson_p": round(float(p_val), 4),
        "sat_mean": round(float(sat.mean()), 3),
        "station_mean": round(float(station.mean()), 3),
    }


# ---------------------------------------------------------------------------
# 4.3 Cloud bias diagnostics
# ---------------------------------------------------------------------------

def compute_cloud_free_fraction(
    lst_df: pd.DataFrame,
    expected_obs_per_month: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Compute cloud-free fraction by site/sensor/month/day-night.
    
    Expected observations per month by sensor:
      MODIS Terra/Aqua: ~30 (daily), realistic clear: ~10-20
      Landsat 8/9: ~2 (16-day), realistic clear: ~1-2
    """
    if expected_obs_per_month is None:
        expected_obs_per_month = {
            "MODIS_Terra": 30,
            "MODIS_Aqua": 30,
            "Landsat_8": 2,
            "Landsat_9": 2,
            "ECOSTRESS": 4,
            "VIIRS": 30,
        }

    if lst_df.empty:
        return pd.DataFrame()

    # Count actual observations per site/sensor/month/day-night/zone
    if "year_month" not in lst_df.columns and "obs_date" in lst_df.columns:
        lst_df = lst_df.copy()
        lst_df["year_month"] = pd.to_datetime(lst_df["obs_date"]).dt.to_period("M")

    group_cols = ["site_id", "region_code", "sensor_id", "time_of_day"]
    if "year_month" in lst_df.columns:
        group_cols.append("year_month")
    if "zone_name" in lst_df.columns:
        group_cols.append("zone_name")

    counts = lst_df.groupby(group_cols).agg(
        n_obs=("lst_c", "count"),
    ).reset_index()

    # Add expected and compute fraction
    counts["expected_obs"] = counts["sensor_id"].map(expected_obs_per_month).fillna(15)
    counts["cloud_free_fraction"] = (counts["n_obs"] / counts["expected_obs"]).clip(0, 1)

    return counts


def compute_cloud_bias_regression(
    delta_t_df: pd.DataFrame,
    cloud_fractions: pd.DataFrame,
) -> dict:
    """
    Test whether Delta-T correlates with cloud-free fraction.
    If significant, suggests clear-sky sampling bias.
    
    Delta-T ~ cloud_free_fraction + region FE + month FE
    """
    if delta_t_df.empty or cloud_fractions.empty:
        return {"status": "insufficient_data"}

    # Aggregate cloud fraction to site-month level
    cf_agg = cloud_fractions.groupby(["site_id", "sensor_id"]).agg(
        mean_cloud_free=("cloud_free_fraction", "mean"),
    ).reset_index()

    # Merge with Delta-T
    dt_agg = delta_t_df.groupby(["site_id"]).agg(
        mean_delta_t=("delta_t_day_c", "mean"),
        region_code=("region_code", "first"),
    ).reset_index()

    merged = dt_agg.merge(
        cf_agg.groupby("site_id")["mean_cloud_free"].mean().reset_index(),
        on="site_id", how="inner"
    )

    if len(merged) < 5:
        return {"status": "insufficient_overlap", "n": len(merged)}

    try:
        from scipy.stats import linregress
        slope, intercept, r_val, p_val, se = linregress(
            merged["mean_cloud_free"], merged["mean_delta_t"]
        )
    except ValueError:
        # All x values identical — no variance to regress on
        return {
            "status": "uninformative",
            "n_sites": len(merged),
            "reason": "Cloud-free fractions are uniform — regression requires varied coverage (multi-year data needed)",
            "mean_cloud_free": round(float(merged["mean_cloud_free"].mean()), 4),
        }

    return {
        "status": "complete",
        "n_sites": len(merged),
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r_val ** 2), 4),
        "p_value": round(float(p_val), 4),
        "is_significant": p_val < 0.05,
        "interpretation": (
            "Cloud fraction IS a significant predictor of Delta-T (p<0.05). "
            "This suggests potential clear-sky sampling bias."
            if p_val < 0.05 else
            "Cloud fraction is NOT a significant predictor of Delta-T (p≥0.05). "
            "No evidence of clear-sky sampling bias."
        ),
    }


def compute_differential_missingness(
    lst_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check if treated (footprint) and control (control_far) zones
    have different cloud-free fractions. Differential missingness
    would bias Delta-T estimates.
    """
    if lst_df.empty or "zone_name" not in lst_df.columns:
        return pd.DataFrame()

    # Treated vs control zone classification
    treated_zones = {"footprint", "near"}
    control_zones = {"control_near", "control_far"}

    df = lst_df.copy()
    df["zone_type"] = df["zone_name"].apply(
        lambda z: "treated" if z in treated_zones
        else "control" if z in control_zones
        else "buffer"
    )

    # Count obs per site/sensor/month/zone_type
    counts = df.groupby(["site_id", "sensor_id", "zone_type"]).agg(
        n_obs=("lst_c", "count"),
    ).reset_index()

    # Pivot to compare treated vs control
    pivot = counts.pivot_table(
        index=["site_id", "sensor_id"],
        columns="zone_type",
        values="n_obs",
        fill_value=0,
    ).reset_index()

    if "treated" in pivot.columns and "control" in pivot.columns:
        pivot["obs_ratio"] = pivot["treated"] / pivot["control"].clip(lower=1)
        pivot["is_differential"] = (pivot["obs_ratio"] < 0.8) | (pivot["obs_ratio"] > 1.25)

    return pivot


# ---------------------------------------------------------------------------
# Full validation runner
# ---------------------------------------------------------------------------

def run_validation_suite(year: int = 2024) -> dict:
    """
    Run full Stage 4 validation:
      1. Find nearby weather stations
      2. Fetch station temperatures
      3. Compute anomalies and compare with satellite
      4. Cloud bias diagnostics
    """
    client = bigquery.Client(project=GCP_PROJECT)

    # Load data
    sites_df = client.query(
        f"SELECT * FROM `{GCP_PROJECT}.dctii_ref.site_registry`"
    ).to_dataframe()

    delta_t_df = client.query(f"""
        SELECT * FROM `{GCP_PROJECT}.dctii_curated.delta_t_monthly`
        WHERE year = {year}
    """).to_dataframe()

    lst_df = client.query(f"""
        SELECT site_id, region_code, sensor_id, time_of_day, zone_name,
               obs_date, lst_c
        FROM `{GCP_PROJECT}.dctii_staging.lst_observations`
        WHERE EXTRACT(YEAR FROM obs_date) = {year}
    """).to_dataframe()

    results = {"year": year}

    # 1. Find nearby stations
    logger.info("Finding nearby weather stations...")
    stations = find_nearby_stations(sites_df, max_distance_km=10.0)
    results["n_station_pairs"] = len(stations)
    results["n_sites_with_stations"] = stations["site_id"].nunique() if not stations.empty else 0

    if not stations.empty:
        # Select best station per site (closest non-airport)
        best_stations = stations.groupby("site_id").first().reset_index()
        results["best_stations"] = best_stations[
            ["site_id", "station_name", "distance_km", "is_airport"]
        ].to_dict("records")[:10]  # Show first 10

        # 2. Fetch station temps
        logger.info("Fetching station temperatures...")
        usaf_wban = list(zip(
            best_stations["usaf"].astype(str),
            best_stations["wban"].astype(str),
        ))

        # Also find baseline stations (>10km from any DC)
        baseline = find_nearby_stations(sites_df, max_distance_km=50.0)
        baseline = baseline[baseline["distance_km"] > 15.0]
        baseline_best = baseline.groupby("region").first().reset_index()

        all_stations = usaf_wban + list(zip(
            baseline_best["usaf"].astype(str),
            baseline_best["wban"].astype(str),
        ))

        station_temps = fetch_station_temps(all_stations, year=year)

        if not station_temps.empty:
            # 3. Compute anomalies
            logger.info("Computing station anomalies...")
            anomalies = compute_station_anomalies(
                station_temps, best_stations, baseline_best
            )

            # 4. Compare with satellite
            logger.info("Comparing with satellite Delta-T...")
            val_result = validate_against_stations(
                delta_t_df, anomalies, best_stations
            )
            results["validation"] = val_result

    # 5. Cloud bias diagnostics
    logger.info("Running cloud bias diagnostics...")
    cloud_fracs = compute_cloud_free_fraction(lst_df)
    results["cloud_free_summary"] = {
        "mean_fraction": round(float(cloud_fracs["cloud_free_fraction"].mean()), 3)
        if not cloud_fracs.empty else None,
        "n_records": len(cloud_fracs),
    }

    cloud_bias = compute_cloud_bias_regression(delta_t_df, cloud_fracs)
    results["cloud_bias"] = cloud_bias

    # 6. Differential missingness
    diff_miss = compute_differential_missingness(lst_df)
    if not diff_miss.empty and "is_differential" in diff_miss.columns:
        n_diff = int(diff_miss["is_differential"].sum())
        results["differential_missingness"] = {
            "n_site_sensors": len(diff_miss),
            "n_differential": n_diff,
            "fraction_differential": round(n_diff / max(len(diff_miss), 1), 3),
        }

    logger.info("Validation suite complete")
    return results
