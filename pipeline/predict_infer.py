"""
DCTII-Predict — Inference Utilities (ML Module 6)

Provides feature extraction, climate zone lookup, PUE imputation,
distribution shift detection, and regional fallback logic for the
online prediction endpoint.

Reuses:
  - pipeline/ancillary_data.py GEE collections and band logic
  - pipeline/indicator_compute.py waste heat flux formula
  - pipeline/dctii_calculator.py normalization and scoring
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("dctii.predict_infer")

GCP_PROJECT = os.environ.get("GCP_PROJECT", "oil-tank-monitoring-123")


# ---------------------------------------------------------------------------
# 1. GEE Feature Extraction at Query Time
# ---------------------------------------------------------------------------

def extract_predict_features(lat: float, lon: float, year: int = None) -> dict:
    """
    Extract biophysical site features for a candidate DC location.
    Reuses ancillary_data.py GEE collections and band logic.
    Returns a dict matching partial FEATURE_COLUMNS (DC specs filled by user).
    """
    import ee

    if year is None:
        year = datetime.now(timezone.utc).year - 1

    ee.Initialize()

    point = ee.Geometry.Point([lon, lat])
    buffer = point.buffer(500)  # 500m — consistent with DCTII footprint zone

    # NDVI: Landsat growing-season composite (Apr–Sep)
    landsat = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterDate(f"{year}-04-01", f"{year}-09-30")
        .filterBounds(buffer)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .map(lambda img: img.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI"))
    )
    ndvi_max = (
        landsat.reduce(ee.Reducer.max())
        .reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=30, maxPixels=1e6
        )
        .get("NDVI_max")
        .getInfo()
    ) or 0.3

    # Impervious fraction: ESA WorldCover 2021
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
    built_up = worldcover.eq(50)
    impervious = (
        built_up.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=10, maxPixels=1e6
        )
        .get("Map")
        .getInfo()
    ) or 0.5

    tree_cover = (
        worldcover.eq(10)
        .reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=10, maxPixels=1e6
        )
        .get("Map")
        .getInfo()
    ) or 0.1

    bare_frac = (
        worldcover.eq(60)
        .reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=10, maxPixels=1e6
        )
        .get("Map")
        .getInfo()
    ) or 0.1

    # Population density: WorldPop 100m
    worldpop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("year", year))
        .filterBounds(buffer)
        .mosaic()
    )
    pop_density = (
        worldpop.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=100, maxPixels=1e6
        )
        .get("population")
        .getInfo()
    ) or 500

    # Elevation: SRTM 30m
    srtm = ee.Image("USGS/SRTMGL1_003")
    elevation = (
        srtm.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=30, maxPixels=1e6
        )
        .get("elevation")
        .getInfo()
    ) or 200

    # Snow cover days: MODIS MOD10A2
    snow = (
        ee.ImageCollection("MODIS/061/MOD10A2")
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filterBounds(buffer)
        .map(lambda img: img.select("Snow_Cover_8Day").gt(0))
    )
    snow_days = (
        snow.reduce(ee.Reducer.sum())
        .reduceRegion(
            reducer=ee.Reducer.mean(), geometry=buffer, scale=500, maxPixels=1e6
        )
        .get("Snow_Cover_8Day_sum")
        .getInfo()
    ) or 0
    snow_days_annual = int(snow_days * 8)  # 8-day composites → days

    return {
        "ndvi_growing_max": float(ndvi_max),
        "impervious_fraction": float(impervious),
        "tree_cover_fraction": float(tree_cover),
        "bare_fraction": float(bare_frac),
        "population_density": float(pop_density),
        "elevation_m": float(elevation),
        "snow_cover_days": float(snow_days_annual),
        "extraction_year": year,
        "gee_status": "ok",
    }


# ---------------------------------------------------------------------------
# 2. Climate Zone Lookup
# ---------------------------------------------------------------------------

# Static Köppen-Geiger classification based on lat/lon ranges
# Approximation for North American DC regions (avoids rasterio dependency at API layer)
KOPPEN_ZONES = [
    # (lat_min, lat_max, lon_min, lon_max, zone)
    (31.0, 35.0, -113.0, -110.0, "BWh"),   # Phoenix area
    (29.0, 32.0, -100.0, -96.0, "BSk"),    # Central Texas
    (28.0, 31.0, -96.0, -94.0, "Cfa"),     # Houston
    (37.0, 40.0, -78.0, -76.0, "Cfa"),     # Northern Virginia
    (42.5, 45.0, -80.0, -78.0, "Dfa"),     # Toronto area
    (44.5, 47.0, -74.5, -72.5, "Dfb"),     # Montreal area
]


def lookup_climate_zone(lat: float, lon: float) -> str:
    """
    Look up Köppen-Geiger climate zone for a location.
    Uses bounding-box approximation for North American DC regions.
    Falls back to 'Cfa' (humid subtropical) if no match.
    """
    for lat_min, lat_max, lon_min, lon_max, zone in KOPPEN_ZONES:
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return zone
    # Latitude-based fallback for North America
    if lat > 50:
        return "Dfb"
    elif lat > 43:
        return "Dfa"
    elif lat > 35:
        return "Cfa"
    elif lat > 30:
        return "BSk"
    else:
        return "BWh"


def lookup_climate_zone_raster(lat: float, lon: float) -> str:
    """
    Look up Köppen-Geiger climate zone from Beck et al. 2018 raster.
    Requires rasterio and the GCS-hosted raster file.
    Falls back to bounding-box method if raster unavailable.
    """
    KOPPEN_CLASS_MAP = {
        4: "BWh", 7: "BSk", 14: "Cfa", 26: "Dfa", 27: "Dfb",
    }
    try:
        import rasterio
        gcs_path = os.environ.get(
            "KOPPEN_RASTER_PATH",
            f"gs://dctii-raw-dev/static/koppen_beck2018_1km.tif",
        )
        with rasterio.open(gcs_path) as src:
            row, col = src.index(lon, lat)
            value = src.read(1)[row, col]
        return KOPPEN_CLASS_MAP.get(int(value), "Cfa")
    except Exception as e:
        logger.warning(f"Raster lookup failed ({e}), using bounding-box fallback")
        return lookup_climate_zone(lat, lon)


# ---------------------------------------------------------------------------
# 3. PUE Imputation
# ---------------------------------------------------------------------------

# Simple regression-based PUE imputation from site characteristics
# Fitted on DCTII site_registry tier-1 sites
PUE_INTERCEPT = 1.45
PUE_COEFS = {
    "capacity_mw": -0.0008,     # larger DCs tend to be more efficient
    "activation_year_norm": -0.015,  # newer DCs are more efficient
    "climate_heat_rank": 0.02,  # hotter climates need more cooling
}
PUE_BY_COOLING = {
    "air_cooled": 1.25,
    "tower_cooled": 1.18,
    "unknown": 1.30,
}


def impute_pue(
    climate_zone: str,
    capacity_mw: float,
    activation_year: int,
    cooling_type: str = "unknown",
) -> float:
    """
    Estimate PUE when not provided by user.
    Uses a simple linear model fitted on tier-1 DCTII sites.
    """
    from pipeline.predict_train import CLIMATE_ZONE_HEAT_RANK

    base = PUE_BY_COOLING.get(cooling_type, 1.30)
    heat_rank = CLIMATE_ZONE_HEAT_RANK.get(climate_zone, 3)
    year_norm = (activation_year - 2020) if activation_year else 0

    pue = base + (
        PUE_COEFS["capacity_mw"] * capacity_mw
        + PUE_COEFS["activation_year_norm"] * year_norm
        + PUE_COEFS["climate_heat_rank"] * heat_rank
    )
    return round(max(1.05, min(2.5, pue)), 2)


# ---------------------------------------------------------------------------
# 4. Regional Fallback Features (GEE timeout)
# ---------------------------------------------------------------------------

# Regional median biophysical features from training data
REGIONAL_FALLBACKS = {
    "BWh": {  # Phoenix
        "ndvi_growing_max": 0.15, "impervious_fraction": 0.45,
        "tree_cover_fraction": 0.02, "bare_fraction": 0.35,
        "population_density": 320.0, "elevation_m": 340.0, "snow_cover_days": 0,
    },
    "BSk": {  # Central Texas
        "ndvi_growing_max": 0.35, "impervious_fraction": 0.30,
        "tree_cover_fraction": 0.15, "bare_fraction": 0.20,
        "population_density": 210.0, "elevation_m": 200.0, "snow_cover_days": 0,
    },
    "Cfa": {  # Houston / NoVA
        "ndvi_growing_max": 0.55, "impervious_fraction": 0.50,
        "tree_cover_fraction": 0.20, "bare_fraction": 0.05,
        "population_density": 750.0, "elevation_m": 80.0, "snow_cover_days": 5,
    },
    "Dfa": {  # Toronto
        "ndvi_growing_max": 0.50, "impervious_fraction": 0.55,
        "tree_cover_fraction": 0.15, "bare_fraction": 0.05,
        "population_density": 850.0, "elevation_m": 175.0, "snow_cover_days": 60,
    },
    "Dfb": {  # Montreal
        "ndvi_growing_max": 0.45, "impervious_fraction": 0.40,
        "tree_cover_fraction": 0.25, "bare_fraction": 0.05,
        "population_density": 720.0, "elevation_m": 35.0, "snow_cover_days": 90,
    },
}

# Default fallback when climate zone is unknown
DEFAULT_FALLBACK = {
    "ndvi_growing_max": 0.35, "impervious_fraction": 0.45,
    "tree_cover_fraction": 0.15, "bare_fraction": 0.10,
    "population_density": 500.0, "elevation_m": 200.0, "snow_cover_days": 15,
}


def get_regional_fallback_features(lat: float, lon: float) -> dict:
    """
    Return median biophysical features for the climate zone when GEE times out.
    """
    climate_zone = lookup_climate_zone(lat, lon)
    features = REGIONAL_FALLBACKS.get(climate_zone, DEFAULT_FALLBACK).copy()
    features["extraction_year"] = datetime.now(timezone.utc).year - 1
    features["gee_status"] = "fallback_used"
    return features


# ---------------------------------------------------------------------------
# 5. Footprint Estimation
# ---------------------------------------------------------------------------

def estimate_footprint(capacity_mw: float) -> float:
    """
    Estimate building footprint from nameplate capacity.
    Linear regression from site_registry: footprint ≈ 0.0015 * capacity + 0.02 km²
    """
    return max(0.01, 0.0015 * capacity_mw + 0.02)


# ---------------------------------------------------------------------------
# Day Prediction Routing (CEM vs Ring models)
# ---------------------------------------------------------------------------

def predict_day(
    models: dict,
    X: np.ndarray,
    site_context: dict,
) -> Tuple[float, float, float, str]:
    """
    Route day prediction to appropriate model based on site context.

    Primary model (day_cem): used for all new/unknown locations.
    Ring model (day_ring): used only for confirmed NOVA cluster sites
    where CEM matching is known to be unavailable.

    For prospective planning (the main use case), day_cem is ALWAYS used
    because the query site has no estimation history.

    Args:
        models: dict from load_model_artifacts() containing day_cem_* and day_ring_* models
        X: feature array (1, n_features)
        site_context: dict with optional 'is_cluster_site', 'region_hint', 'climate_heat_rank'

    Returns:
        (median, lower_ci, upper_ci, method_used)
    """
    from pipeline.predict_train import apply_bias_offset

    is_nova_cluster = (
        site_context.get("is_cluster_site", False)
        and site_context.get("region_hint", "").upper() == "NOVA"
    )

    corrections = models.get("corrections", {})
    climate_rank = int(site_context.get("climate_heat_rank", 3))

    if is_nova_cluster:
        model_key = "day_ring"
        discount = 0.4
        method_used = "ring_difference_discounted"
        corr = corrections.get("day_ring_correction", 0.0)
    else:
        model_key = "day_cem"
        discount = 1.0
        method_used = "cem_primary"
        corr = corrections.get("day_cem_correction", 0.0)

    med_raw = models[f"{model_key}_median"].predict(X)[0] * discount
    q10 = models[f"{model_key}_q10"].predict(X)[0] * discount
    q90 = models[f"{model_key}_q90"].predict(X)[0] * discount

    # Apply stratified bias correction
    bias_offsets = corrections.get("day_cem_bias_offsets", {})
    med = apply_bias_offset(med_raw, climate_rank, bias_offsets)

    return (
        float(max(0.0, med)),
        float(max(0.0, q10 - corr)),
        float(max(0.0, q90 + corr)),
        method_used,
    )


# ---------------------------------------------------------------------------
# Standalone helper functions (extracted for testability)
# ---------------------------------------------------------------------------

def resolve_pue(req, climate_zone: str) -> Tuple[float, bool]:
    """Resolve PUE: use provided value or impute from climate/capacity."""
    if req.pue is not None:
        return req.pue, False
    pue = impute_pue(
        climate_zone=climate_zone,
        capacity_mw=req.capacity_mw,
        activation_year=datetime.now(timezone.utc).year,
        cooling_type=req.cooling_type if isinstance(req.cooling_type, str)
                     else req.cooling_type.value,
    )
    return pue, True


def resolve_footprint(req) -> Tuple[float, bool]:
    """Resolve footprint: use provided value or estimate from capacity."""
    if req.footprint_km2 is not None:
        return req.footprint_km2, False
    return max(0.01, 0.0015 * req.capacity_mw + 0.02), True


# DCTII score composition
NORM_BOUNDS = {
    "delta_t_day": (0, 3),
    "delta_t_night": (0, 2),
    "heat_island_area_km2": (0, 10),
    "population_exposed": (0, 50000),
    "waste_heat_flux_wm2": (0, 250),
}
WEIGHT_SCHEMES = {
    "expert": [0.25, 0.30, 0.10, 0.10, 0.25],
    "equal": [0.20, 0.20, 0.20, 0.20, 0.20],
    "pca": [0.22, 0.28, 0.12, 0.12, 0.26],
    "entropy": [0.24, 0.29, 0.11, 0.11, 0.25],
}


def compose_dctii_score(
    dt_day: float, dt_night: float, heat_area: float,
    pop_exposed: float, waste_heat: float, scheme: str = "expert",
) -> Tuple[float, str]:
    """Compute composite DCTII score (0–100) with impact category."""
    from pipeline.dctii_calculator import assign_impact_category

    def _norm(v, lo, hi):
        return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else 0.0

    indicators = [
        _norm(dt_day,      *NORM_BOUNDS["delta_t_day"]),
        _norm(dt_night,    *NORM_BOUNDS["delta_t_night"]),
        _norm(heat_area,   *NORM_BOUNDS["heat_island_area_km2"]),
        _norm(pop_exposed, *NORM_BOUNDS["population_exposed"]),
        _norm(waste_heat,  *NORM_BOUNDS["waste_heat_flux_wm2"]),
    ]
    w = WEIGHT_SCHEMES.get(scheme, WEIGHT_SCHEMES["expert"])
    score = round(sum(v * wt for v, wt in zip(indicators, w)) * 100, 1)
    return score, assign_impact_category(score)


# Climate-stratified GEE fallback defaults
ERA5_DEFAULTS_BY_CLIMATE = {
    "BWh": {"ndvi_growing_max": 0.15, "impervious_fraction": 0.65,
            "tree_cover_fraction": 0.05, "bare_fraction": 0.30,
            "population_density": 320, "elevation_m": 340, "snow_cover_days": 0},
    "BSk": {"ndvi_growing_max": 0.25, "impervious_fraction": 0.55,
            "tree_cover_fraction": 0.10, "bare_fraction": 0.20,
            "population_density": 210, "elevation_m": 280, "snow_cover_days": 5},
    "Cfa": {"ndvi_growing_max": 0.45, "impervious_fraction": 0.60,
            "tree_cover_fraction": 0.15, "bare_fraction": 0.10,
            "population_density": 750, "elevation_m": 120, "snow_cover_days": 0},
    "Dfa": {"ndvi_growing_max": 0.50, "impervious_fraction": 0.55,
            "tree_cover_fraction": 0.20, "bare_fraction": 0.08,
            "population_density": 850, "elevation_m": 150, "snow_cover_days": 20},
    "Dfb": {"ndvi_growing_max": 0.55, "impervious_fraction": 0.50,
            "tree_cover_fraction": 0.25, "bare_fraction": 0.05,
            "population_density": 720, "elevation_m": 50, "snow_cover_days": 45},
}


def get_climate_stratified_defaults(climate_zone: str) -> dict:
    """Return climate-stratified fallback features when GEE is unavailable."""
    defaults = ERA5_DEFAULTS_BY_CLIMATE.get(
        climate_zone, ERA5_DEFAULTS_BY_CLIMATE["Cfa"]
    )
    return {**defaults, "gee_status": "climate_fallback"}


# Day/night scaling factors by climate zone
_DAY_NIGHT_RATIO = {
    "BWh": 1.8,   # arid: strong daytime insolation amplifies UHI
    "BSk": 1.5,
    "Cfa": 1.3,   # humid subtropical: moderate day amplification
    "Dfa": 1.2,
    "Dfb": 1.1,   # cold continental: minimal day/night difference
}


def derive_day_from_night(dt_night: float, climate_zone: str) -> float:
    """Derive daytime ΔT from nighttime ΔT using climate-specific ratio."""
    ratio = _DAY_NIGHT_RATIO.get(climate_zone, 1.3)
    return max(0.0, dt_night * ratio)


def spearman_confidence_interval(
    rho: float, n: int, alpha: float = 0.05
) -> Tuple[float, float]:
    """Fisher z-transformation CI for Spearman rho. Valid for n >= 10."""
    from scipy import stats
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_lo = z - z_crit * se
    z_hi = z + z_crit * se
    return float(np.tanh(z_lo)), float(np.tanh(z_hi))


def check_cool_island_risk(
    climate_zone: str,
    ndvi: float,
    impervious: float,
) -> bool:
    """
    Flag sites that may be cool islands due to irrigation in arid climates.
    PHX_002 and PHX_006 both match this pattern.
    Returns True if site may exhibit evaporative cooling that reverses DeltaT sign.
    """
    arid_climates = {"BWh", "BSk", "BSh"}
    return (
        climate_zone in arid_climates
        and ndvi > 0.30
        and impervious < 0.55
    )
