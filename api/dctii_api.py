"""
DCTII REST API — Data Center Thermal Impact Index
FastAPI service deployed on Cloud Run (mirrors GOII pattern).
Serves DCTII scores, indicators, and site metadata from BigQuery.

Security:
  - Public endpoints: /health, /api/v1/map-sites, /api/v1/meta (read-only, no secrets)
  - Protected endpoints: all others require X-API-Key header (dev mode: no key needed)
  - Rate limiting: 100 requests/minute per API key
  - All queries use parameterized BigQuery parameters (no f-string SQL injection)
  - CORS restricted to known origins in production
"""

import os
import time
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional, List

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator, model_validator
from enum import Enum

from fastapi import FastAPI, HTTPException, Query, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from google.cloud import bigquery

# Predict pipeline imports (lazy-safe — modules exist but GCS/BQ not needed at import)
from pipeline.predict_train import (
    engineer_features, FEATURE_COLUMNS, apply_bias_offset,
    CLIMATE_ZONE_HEAT_RANK, compute_distribution_shift,
)
from pipeline.predict_infer import (
    extract_predict_features, lookup_climate_zone, impute_pue,
    predict_day, resolve_pue, compose_dctii_score,
    get_climate_stratified_defaults, check_cool_island_risk,
)
from pipeline.dctii_calculator import assign_impact_category

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dctii-api")

PROJECT_ID = os.getenv("GCP_PROJECT", "oil-tank-monitoring-123")
SERVING_DATASET = os.getenv("BQ_SERVING_DATASET", "dctii_serving")
REF_DATASET = os.getenv("BQ_REF_DATASET", "dctii_ref")
CURATED_DATASET = os.getenv("BQ_CURATED_DATASET", "dctii_curated")
API_KEYS = set(os.getenv("DCTII_API_KEYS", "").split(","))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# M-08: Single version constant across API, health, docs
API_VERSION = os.getenv("API_VERSION", "1.1.0")

# ── Score Scaling ────────────────────────────────────────────────────────────
# Raw composite scores are dominated by waste_heat_flux (theoretical) while
# delta_t_night, heat_island_area, and population_exposed are null/zero in
# the pilot dataset.  We recompute a physically-anchored score at the API
# layer using delta-T as the primary signal (70 %) and waste-heat as a
# secondary modifier (30 %).  Negative delta-T (cooler than control) is
# clamped to zero — no credit for cooling artefacts.
#
# Reference thresholds (physical, not sample-relative):
#   delta-T  1.5 °C  → score component saturates at 100 %
#   waste-heat 200 W/m²  → score component saturates at 100 %

_DT_REF = 1.5    # °C – delta-T at which thermal component maxes out
_WH_REF = 200.0  # W/m² – waste heat flux reference ceiling

def _recompute_score(site: dict) -> float:
    """Physically-anchored 0-100 score from available indicators."""
    dt_day = max(site.get("delta_t_day") or 0.0, 0.0)
    dt_night = max(site.get("delta_t_night") or 0.0, 0.0)
    dt = max(dt_day, dt_night)  # strongest measured signal
    wh = max(site.get("waste_heat_flux_wm2") or 0.0, 0.0)
    hi = max(site.get("heat_island_area_km2") or 0.0, 0.0)
    pop = max(site.get("population_exposed") or 0.0, 0.0)

    dt_norm = min(dt / _DT_REF, 1.0)
    wh_norm = min(wh / _WH_REF, 1.0)
    # When hi/pop become available they'll contribute; for now they're 0
    hi_norm = min(hi / 50.0, 1.0) if hi > 0 else 0.0
    pop_norm = min(pop / 100000.0, 1.0) if pop > 0 else 0.0

    # Weights: delta-T anchors the score; waste heat amplifies
    score = (0.55 * dt_norm + 0.25 * wh_norm
             + 0.10 * hi_norm + 0.10 * pop_norm) * 100.0
    return round(min(score, 100.0), 1)

def _impact_category(score_100):
    """Map a 0-100 score to impact tier."""
    if score_100 is None:
        return None
    if score_100 >= 80:
        return "Severe"
    if score_100 >= 60:
        return "High"
    if score_100 >= 40:
        return "Moderate"
    if score_100 >= 20:
        return "Low"
    return "Minimal"

app = FastAPI(
    title="DCTII API",
    description="Data Center Thermal Impact Index — scores, indicators, and site metadata",
    version=API_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# H-04: Lazy BQ client init — avoids import-time crashes on Cloud Run
_bq_client: Optional[bigquery.Client] = None


def get_bq_client() -> bigquery.Client:
    """Get or create BigQuery client (lazy init, H-04)."""
    global _bq_client
    if _bq_client is None:
        try:
            _bq_client = bigquery.Client(project=PROJECT_ID)
        except Exception as e:
            logger.error(f"BigQuery client init failed: {e}")
            raise HTTPException(503, "Data service unavailable")
    return _bq_client


# --- API Key Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
# H-03: _rate_counts is in-memory and per-instance.
# On Cloud Run with min_instances > 1 or under autoscale,
# the effective rate limit is RATE_LIMIT_PER_MINUTE * num_instances.
# Replace with Firestore or Redis if strict per-key limits are required.
_rate_counts: dict = defaultdict(list)


def _check_rate_limit(api_key: str):
    now = time.time()
    window = [t for t in _rate_counts[api_key] if now - t < 60]
    _rate_counts[api_key] = window
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(429, f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} req/min)")
    _rate_counts[api_key].append(now)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if not API_KEYS or API_KEYS == {""}:
        return "dev"
    if not api_key or api_key not in API_KEYS:
        raise HTTPException(401, "Invalid or missing API key")
    _check_rate_limit(api_key)
    return api_key


# ── Helper ───────────────────────────────────────────────────────────────────

def _bq_query(sql: str, params: list = None) -> list:
    cfg = bigquery.QueryJobConfig(query_parameters=params or [])
    return [dict(r) for r in get_bq_client().query(sql, job_config=cfg).result()]


def _serialize(rows: list, scale_scores: bool = False) -> list:
    """Convert BQ Row dicts to JSON-safe dicts (handle date/datetime).
    If scale_scores=True, recompute dctii_score on 0-100 scale using
    physically-anchored formula and derive impact_category."""
    import datetime as dt
    out = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if isinstance(v, (dt.date, dt.datetime)):
                clean[k] = v.isoformat()
            else:
                clean[k] = v
        if scale_scores and clean.get("dctii_score") is not None:
            clean["dctii_score"] = _recompute_score(clean)
            clean["impact_category"] = _impact_category(clean["dctii_score"])
            # H-06: CI bounds set to None pending proper uncertainty propagation.
            # The previous ±15% relative band was statistically invalid.
            clean["ci_lower"] = None
            clean["ci_upper"] = None
            clean["ci_method"] = "pending_propagation"
        out.append(clean)
    return out


# ── Public Endpoints (no auth) ───────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check with BQ connectivity and predict model status (H-05)."""
    bq_status = "unknown"
    try:
        client = get_bq_client()
        client.query("SELECT 1").result(timeout=5)
        bq_status = "connected"
    except Exception as e:
        logger.error(f"Health check BQ probe failed: {e}")
        bq_status = str(e)

    predict_status = "loaded" if _predict_models is not None else "not_loaded"
    predict_version = None
    if _predict_models is not None:
        predict_version = _predict_models.get("eval_report", {}).get("version")

    overall = "ok" if bq_status == "connected" else "degraded"
    resp = {
        "status": overall,
        "bq": bq_status,
        "predict_models": predict_status,
        "predict_version": predict_version,
        "service": "dctii-api",
        "version": API_VERSION,
    }
    if overall == "degraded":
        return JSONResponse(resp, status_code=503)
    return resp


@app.get("/api/v1/meta")
def get_meta():
    """API metadata: available years, data versions, site count."""
    scores_meta = _bq_query(f"""
        SELECT
            COUNT(DISTINCT site_id) AS site_count,
            COUNT(DISTINCT year) AS year_count,
            ARRAY_AGG(DISTINCT year ORDER BY year) AS years,
            MAX(data_version) AS latest_version,
            ARRAY_AGG(DISTINCT weighting_scheme) AS weighting_schemes
        FROM `{PROJECT_ID}.{SERVING_DATASET}.dctii_scores`
    """)
    meta = scores_meta[0] if scores_meta else {}
    return {
        "project": "DCTII — Data Center Thermal Impact Index",
        "site_count": meta.get("site_count", 0),
        "years": meta.get("years", []),
        "latest_data_version": meta.get("latest_version"),
        "weighting_schemes": meta.get("weighting_schemes", ["equal"]),
        "regions": ["PHX", "HOU", "NOVA", "CTX", "TOR", "MTL"],
    }


@app.get("/api/v1/map-sites")
def get_map_sites(
    year: Optional[int] = Query(None, description="Filter scores to specific year"),
    weighting: str = Query("equal", description="Weighting scheme"),
):
    """Single payload for dashboard map: site metadata + scores joined."""
    year_filter = "AND s.year = @year" if year else ""
    params = [bigquery.ScalarQueryParameter("weighting", "STRING", weighting)]
    if year:
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))

    rows = _bq_query(f"""
        SELECT
            r.site_id, r.region_code, r.latitude, r.longitude,
            r.operator, r.capacity_mw, r.pue_estimate, r.cooling_type,
            r.activation_year, r.country, r.climate_zone,
            s.year, s.dctii_score, s.impact_category,
            s.delta_t_day, s.delta_t_night,
            s.heat_island_area_km2, s.population_exposed,
            s.waste_heat_flux_wm2,
            s.ci_lower, s.ci_upper
        FROM `{PROJECT_ID}.{REF_DATASET}.site_registry` r
        LEFT JOIN `{PROJECT_ID}.{SERVING_DATASET}.dctii_scores` s
          ON r.site_id = s.site_id
          AND s.weighting_scheme = @weighting
          {year_filter}
        ORDER BY r.site_id
    """, params)
    return {"sites": _serialize(rows, scale_scores=True), "count": len(rows)}


# ── Protected Endpoints (API key required in production) ─────────────────────

@app.get("/api/v1/sites")
def list_sites(
    country: Optional[str] = Query(None, description="Filter by country code (US, CA)"),
    region: Optional[str] = Query(None, description="Filter by region code"),
    _key: str = Depends(verify_api_key),
):
    """List all data center sites in the registry."""
    sql = f"SELECT * FROM `{PROJECT_ID}.{REF_DATASET}.site_registry` WHERE 1=1"
    params = []
    if country:
        sql += " AND country = @country"
        params.append(bigquery.ScalarQueryParameter("country", "STRING", country))
    if region:
        sql += " AND region_code = @region"
        params.append(bigquery.ScalarQueryParameter("region", "STRING", region))
    sql += " ORDER BY site_id"
    return {"sites": _serialize(_bq_query(sql, params))}


@app.get("/api/v1/scores")
def get_scores(
    site_id: Optional[str] = Query(None),
    year: Optional[int] = Query(None),
    weighting: str = Query("equal"),
    data_version: Optional[int] = Query(None),
    _key: str = Depends(verify_api_key),
):
    """Retrieve DCTII composite scores."""
    sql = f"""
        SELECT site_id, year, weighting_scheme, dctii_score, impact_category,
               delta_t_day, delta_t_night, heat_island_area_km2,
               population_exposed, waste_heat_flux_wm2,
               ci_lower, ci_upper, data_version, norm_bounds_hash
        FROM `{PROJECT_ID}.{SERVING_DATASET}.dctii_scores`
        WHERE weighting_scheme = @weighting
    """
    params = [bigquery.ScalarQueryParameter("weighting", "STRING", weighting)]
    if site_id:
        sql += " AND site_id = @site_id"
        params.append(bigquery.ScalarQueryParameter("site_id", "STRING", site_id))
    if year:
        sql += " AND year = @year"
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))
    if data_version is not None:
        sql += " AND data_version = @data_version"
        params.append(bigquery.ScalarQueryParameter("data_version", "INT64", data_version))
    return {"scores": _serialize(_bq_query(sql, params), scale_scores=True)}


@app.get("/api/v1/indicators/{site_id}")
def get_indicators(
    site_id: str,
    year: Optional[int] = Query(None, description="Filter to specific year"),
    _key: str = Depends(verify_api_key),
):
    """Retrieve per-indicator values for a specific site."""
    sql = f"SELECT * FROM `{PROJECT_ID}.{CURATED_DATASET}.site_indicators` WHERE site_id = @site_id"
    params = [bigquery.ScalarQueryParameter("site_id", "STRING", site_id)]
    if year:
        sql += " AND year = @year"
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))
    rows = _bq_query(sql, params)
    if not rows:
        raise HTTPException(404, f"Site {site_id} not found")
    return {"site_id": site_id, "indicators": _serialize(rows)}


@app.get("/api/v1/delta-t/{site_id}")
def get_delta_t(
    site_id: str,
    year: Optional[int] = Query(None, description="Filter to specific year"),
    time_of_day: Optional[str] = Query(None, description="day or night"),
    _key: str = Depends(verify_api_key),
):
    """Monthly Delta-T time series for a site (for charts)."""
    sql = f"""
        SELECT site_id, EXTRACT(YEAR FROM year_month) AS year, 
               EXTRACT(MONTH FROM year_month) AS month,
               sensor_id, delta_t_day_c, delta_t_night_c, 
               n_clear_days, n_clear_nights, reliability_flag, season
        FROM `{PROJECT_ID}.{CURATED_DATASET}.delta_t_monthly`
        WHERE site_id = @site_id
    """
    params = [bigquery.ScalarQueryParameter("site_id", "STRING", site_id)]
    if year:
        sql += " AND EXTRACT(YEAR FROM year_month) = @year"
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))
    if time_of_day:
        # time_of_day filter not applicable — delta_t_monthly has day+night in same row
        pass
    sql += " ORDER BY year_month"
    rows = _bq_query(sql, params)
    if not rows:
        raise HTTPException(404, f"No Delta-T data for {site_id}")
    return {"site_id": site_id, "delta_t": rows}


@app.get("/api/v1/regions")
def get_regions(
    year: Optional[int] = Query(None, description="Filter to specific year"),
    _key: str = Depends(verify_api_key),
):
    """List regions with site counts and mean scores."""
    # M-01: Year filter to avoid cross-year aggregation
    year_filter = "AND s.year = @year" if year else ""
    params = []
    if year:
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))
    rows = _bq_query(f"""
        SELECT
            r.region_code, r.site_id,
            s.dctii_score, s.delta_t_day, s.delta_t_night,
            s.heat_island_area_km2, s.population_exposed,
            s.waste_heat_flux_wm2,
            r.country
        FROM `{PROJECT_ID}.{REF_DATASET}.site_registry` r
        LEFT JOIN `{PROJECT_ID}.{SERVING_DATASET}.dctii_scores` s
          ON r.site_id = s.site_id AND s.weighting_scheme = 'equal'
          {year_filter}
    """, params)
    scaled = _serialize(rows, scale_scores=True)
    # Aggregate by region
    from collections import defaultdict
    region_map = defaultdict(lambda: {"scores": [], "country": None, "sites": 0})
    for s in scaled:
        rc = s["region_code"]
        region_map[rc]["country"] = s.get("country")
        region_map[rc]["sites"] += 1
        if s.get("dctii_score") is not None:
            region_map[rc]["scores"].append(s["dctii_score"])
    result = []
    for rc, d in region_map.items():
        sc = d["scores"]
        result.append({
            "region_code": rc,
            "site_count": d["sites"],
            "mean_score": round(sum(sc) / len(sc), 2) if sc else None,
            "max_score": round(max(sc), 2) if sc else None,
            "country": d["country"],
        })
    result.sort(key=lambda x: x.get("mean_score") or 0, reverse=True)
    return {"regions": result}


@app.get("/api/v1/summary")
def get_summary(
    year: Optional[int] = Query(None, description="Filter to specific year"),
    weighting: str = Query("equal", description="Weighting scheme"),
    _key: str = Depends(verify_api_key),
):
    """Aggregate statistics: distribution by category, region, top sites."""
    # M-01 + M-05: Year and weighting filters
    params = [bigquery.ScalarQueryParameter("weighting", "STRING", weighting)]
    year_filter = ""
    if year:
        year_filter = "AND s.year = @year"
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))
    all_scores = _bq_query(f"""
        SELECT s.site_id, s.dctii_score, s.impact_category,
               s.delta_t_day, s.delta_t_night,
               s.heat_island_area_km2, s.population_exposed,
               s.waste_heat_flux_wm2
        FROM `{PROJECT_ID}.{SERVING_DATASET}.dctii_scores` s
        WHERE s.weighting_scheme = @weighting
        {year_filter}
        ORDER BY s.dctii_score DESC
    """, params)
    scaled = _serialize(all_scores, scale_scores=True)

    scores_list = [s["dctii_score"] for s in scaled if s.get("dctii_score") is not None]
    cats = [s["impact_category"] for s in scaled if s.get("impact_category")]
    summary = {
        "total_sites": len(scores_list),
        "mean_score": round(sum(scores_list) / len(scores_list), 2) if scores_list else 0,
        "min_score": round(min(scores_list), 2) if scores_list else 0,
        "max_score": round(max(scores_list), 2) if scores_list else 0,
        "n_minimal": cats.count("Minimal"),
        "n_low": cats.count("Low"),
        "n_moderate": cats.count("Moderate"),
        "n_high": cats.count("High"),
        "n_severe": cats.count("Severe"),
        "top_5": scaled[:5],
    }
    return summary


@app.get("/api/v1/causal/{site_id}")
def get_causal_results(
    site_id: str,
    data_version: Optional[int] = Query(None),
    _key: str = Depends(verify_api_key),
):
    """Retrieve DiD causal analysis results for a site."""
    sql = f"SELECT * FROM `{PROJECT_ID}.{SERVING_DATASET}.did_results` WHERE site_id = @site_id"
    params = [bigquery.ScalarQueryParameter("site_id", "STRING", site_id)]
    if data_version is not None:
        sql += " AND data_version = @data_version"
        params.append(bigquery.ScalarQueryParameter("data_version", "INT64", data_version))
    rows = _bq_query(sql, params)
    if not rows:
        raise HTTPException(404, f"No causal results for {site_id}")
    return {"site_id": site_id, "causal": _serialize(rows)}


# ══════════════════════════════════════════════════════════════════════════════
# DCTII-Predict: ML Prediction Endpoint (ML Module 6)
# ══════════════════════════════════════════════════════════════════════════════

class PredictCoolingType(str, Enum):
    air_cooled = "air_cooled"
    tower_cooled = "tower_cooled"
    unknown = "unknown"


class PredictRequest(BaseModel):
    latitude: Optional[float] = Field(None, ge=-90, le=90, examples=[33.4484])
    longitude: Optional[float] = Field(None, ge=-180, le=180, examples=[-112.0740])
    address: Optional[str] = Field(None, examples=["1 Data Center Dr, Ashburn VA"])

    capacity_mw: float = Field(..., gt=0, le=5000, examples=[50.0],
                               description="Nameplate IT capacity in MW")
    cooling_type: PredictCoolingType = Field(PredictCoolingType.air_cooled)
    pue: Optional[float] = Field(None, ge=1.0, le=3.0,
                                 description="PUE estimate. If None, ML imputation used.")
    load_factor: float = Field(0.6, ge=0.1, le=1.0)
    footprint_km2: Optional[float] = Field(None, gt=0,
                                           description="Building footprint in km². If None, estimated.")

    weighting_scheme: str = Field("expert", pattern="^(expert|equal|pca|entropy)$")
    reference_year: Optional[int] = Field(None, ge=2015, le=2035,
                                          description="Year for GEE covariate extraction.")

    @model_validator(mode="after")
    def check_location(self):
        if self.longitude is not None and self.latitude is None:
            raise ValueError("longitude requires latitude")
        if self.latitude is None and self.address is None:
            raise ValueError("Either lat/lon or address must be provided")
        return self


class SHAPContribution(BaseModel):
    feature: str
    value: float
    shap_impact: float


class PredictResponse(BaseModel):
    latitude: float
    longitude: float
    address_resolved: Optional[str] = None
    climate_zone: str

    delta_t_day_c: float
    delta_t_night_c: float
    delta_t_day_ci: list
    delta_t_night_ci: list

    heat_island_area_km2: float
    population_exposed: float
    waste_heat_flux_wm2: float

    dctii_score: float
    impact_category: str

    shap_day_top5: list
    shap_night_top5: list

    distribution_shift_score: float
    distribution_shift_label: str
    pue_was_imputed: bool
    footprint_was_estimated: bool
    gee_extraction_status: str
    cool_island_risk: bool
    day_prediction_method: str
    model_version: str
    prediction_id: str


# Lazy-loaded predict models
_predict_models = None


def get_predict_models():
    global _predict_models
    if _predict_models is None:
        from pipeline.predict_train import load_model_artifacts
        logger.info("Loading DCTII-Predict model artifacts from GCS...")
        _predict_models = load_model_artifacts(version="latest")
        logger.info("Model artifacts loaded.")
    return _predict_models


async def _write_prediction_to_bq(bq_row: dict):
    """Fire-and-forget BQ write for prediction audit trail."""
    try:
        client = get_bq_client()
        table_id = f"{PROJECT_ID}.{SERVING_DATASET}.predictions"
        errors = client.insert_rows_json(table_id, [bq_row])
        if errors:
            logger.warning(f"BQ prediction write errors: {errors}")
    except Exception as e:
        logger.warning(f"Failed to write prediction to BQ: {e}")


@app.post("/api/v1/predict", response_model=PredictResponse,
          summary="Predict thermal impact for a planned data center")
async def predict_thermal_impact(
    req: PredictRequest,
    api_key: str = Depends(verify_api_key),
):
    prediction_id = str(uuid.uuid4())
    models = get_predict_models()

    # ── 1. Resolve location ───────────────────────────────────────────────
    lat, lon = req.latitude, req.longitude
    address_resolved = None
    if lat is None or lon is None:
        try:
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="dctii-predict/1.0", timeout=8)
            location = await asyncio.to_thread(geolocator.geocode, req.address)
            if location is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not geocode address: {req.address}",
                )
            lat, lon = location.latitude, location.longitude
            address_resolved = location.address
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Geocoding failed: {e}")

    # ── 2. Validate jurisdiction ──────────────────────────────────────────
    if not (-170 <= lon <= -50 and 24 <= lat <= 72):
        logger.warning(f"Location ({lat},{lon}) outside trained NA bounding box")

    # ── 3. Climate zone ───────────────────────────────────────────────────
    climate_zone = lookup_climate_zone(lat, lon)

    # ── 4. GEE feature extraction (with climate-stratified fallback) ──────
    gee_status = "ok"
    try:
        gee_features = extract_predict_features(lat, lon, year=req.reference_year)
    except Exception as e:
        logger.warning(f"GEE extraction failed: {e} -- using climate fallback")
        gee_features = get_climate_stratified_defaults(climate_zone)
        gee_status = "climate_fallback"

    # ── 4b. Cool island risk flag ─────────────────────────────────────────
    cool_island_risk = check_cool_island_risk(
        climate_zone,
        gee_features.get("ndvi_growing_max", 0),
        gee_features.get("impervious_fraction", 1.0),
    )
    if cool_island_risk:
        logger.info(f"[predict] Cool island risk flagged for ({lat},{lon}) "
                     f"zone={climate_zone}")

    # ── 5. PUE imputation if not provided ────────────────────────────────
    pue = req.pue
    pue_was_imputed = False
    if pue is None:
        pue = impute_pue(
            climate_zone=climate_zone,
            capacity_mw=req.capacity_mw,
            activation_year=datetime.now(timezone.utc).year,
            cooling_type=req.cooling_type.value,
        )
        pue_was_imputed = True

    # ── 6. Footprint estimation if not provided ───────────────────────────
    footprint = req.footprint_km2
    footprint_was_estimated = False
    if footprint is None:
        footprint = max(0.01, 0.0015 * req.capacity_mw + 0.02)
        footprint_was_estimated = True

    # ── 7. Feature engineering ────────────────────────────────────────────
    raw_row = {
        "capacity_mw": req.capacity_mw,
        "pue_estimate": pue,
        "load_factor": req.load_factor,
        "cooling_type": req.cooling_type.value,
        "footprint_km2": footprint,
        "climate_zone": climate_zone,
        "cluster_id": None,
        "covariate_year_proxy": False,
        "confidence_tier": 2,
        "waste_heat_flux_computed": (
            req.capacity_mw * req.load_factor * (pue - 1.0)
            / max(footprint, 0.001)
        ),
        **{k: v for k, v in gee_features.items()
           if k not in ("extraction_year", "gee_status")},
    }
    row_df = pd.DataFrame([raw_row])
    feat_df = engineer_features(row_df)
    X = feat_df[FEATURE_COLUMNS].values

    climate_rank = int(feat_df["climate_heat_rank"].iloc[0])
    corr = models["corrections"]

    # ── 8. Day prediction (stratified CEM/Ring routing) ───────────────────
    site_context = {
        "is_cluster_site": False,
        "region_hint": "",
        "climate_heat_rank": climate_rank,
    }
    dt_day, dt_day_lo, dt_day_hi, day_method = predict_day(
        models, X, site_context,
    )

    # ── 9. Night prediction (with bias correction) ────────────────────────
    dt_night_raw = float(models["night_median"].predict(X)[0])
    dt_night = apply_bias_offset(
        dt_night_raw, climate_rank,
        corr.get("night_bias_offsets", {}),
    )
    night_corr = corr.get("night_correction", 0.0)
    dt_night_lo = max(0.0, float(models["night_q10"].predict(X)[0]) - night_corr)
    dt_night_hi = float(models["night_q90"].predict(X)[0]) + night_corr

    # Round
    dt_day = round(dt_day, 4)
    dt_night = round(dt_night, 4)
    dt_day_lo = round(dt_day_lo, 4)
    dt_day_hi = round(dt_day_hi, 4)
    dt_night_lo = round(dt_night_lo, 4)
    dt_night_hi = round(dt_night_hi, 4)

    # ── 10. Derive secondary indicators ──────────────────────────────────
    waste_heat_flux = float(feat_df["waste_heat_flux"].iloc[0])
    heat_island_area = min(float(np.exp(0.9 * dt_night + 0.3)), 10.0)
    regional_density = float(gee_features.get("population_density", 500))
    population_exposed = heat_island_area * regional_density

    # ── 11. Compose DCTII score ───────────────────────────────────────────
    dctii_score, impact_cat = compose_dctii_score(
        dt_day=dt_day, dt_night=dt_night, heat_area=heat_island_area,
        pop_exposed=population_exposed, waste_heat=waste_heat_flux,
        scheme=req.weighting_scheme,
    )

    # ── 12. SHAP explanations ─────────────────────────────────────────────
    def top5_shap(explainer, label):
        try:
            shap_vals = explainer.shap_values(X)[0]
            idx = np.argsort(np.abs(shap_vals))[::-1][:5]
            return [
                SHAPContribution(
                    feature=FEATURE_COLUMNS[i],
                    value=float(X[0][i]),
                    shap_impact=float(shap_vals[i]),
                )
                for i in idx
            ]
        except Exception as e:
            logger.warning(f"SHAP failed ({label}): {e}")
            return []

    shap_day_top5 = top5_shap(models["shap_day_cem"], "day")
    shap_night_top5 = top5_shap(models["shap_night"], "night")

    # ── 13. Distribution shift ────────────────────────────────────────────
    dist_score, dist_label = compute_distribution_shift(
        X[0], models["distribution"]
    )

    # ── 14. Write to BQ predictions table ────────────────────────────────
    bq_row = {
        "prediction_id": prediction_id,
        "created_ts": datetime.now(timezone.utc).isoformat(),
        "latitude": lat,
        "longitude": lon,
        "climate_zone": climate_zone,
        "capacity_mw": req.capacity_mw,
        "cooling_type": req.cooling_type.value,
        "pue_used": pue,
        "pue_was_imputed": pue_was_imputed,
        "footprint_km2": footprint,
        "load_factor": req.load_factor,
        "delta_t_day_pred": dt_day,
        "delta_t_night_pred": dt_night,
        "delta_t_day_ci_lower": dt_day_lo,
        "delta_t_day_ci_upper": dt_day_hi,
        "delta_t_night_ci_lower": dt_night_lo,
        "delta_t_night_ci_upper": dt_night_hi,
        "heat_island_area_km2": heat_island_area,
        "population_exposed": population_exposed,
        "waste_heat_flux_wm2": waste_heat_flux,
        "dctii_score": dctii_score,
        "impact_category": impact_cat,
        "distribution_shift_score": dist_score,
        "distribution_shift_label": dist_label,
        "model_version": models["eval_report"].get("version", "unknown"),
        "gee_status": gee_status,
        "weighting_scheme": req.weighting_scheme,
        "cool_island_risk": cool_island_risk,
    }
    asyncio.create_task(_write_prediction_to_bq(bq_row))

    return PredictResponse(
        latitude=lat,
        longitude=lon,
        address_resolved=address_resolved,
        climate_zone=climate_zone,
        delta_t_day_c=dt_day,
        delta_t_night_c=dt_night,
        delta_t_day_ci=[dt_day_lo, dt_day_hi],
        delta_t_night_ci=[dt_night_lo, dt_night_hi],
        heat_island_area_km2=round(heat_island_area, 3),
        population_exposed=round(population_exposed, 1),
        waste_heat_flux_wm2=round(waste_heat_flux, 2),
        dctii_score=dctii_score,
        impact_category=impact_cat,
        shap_day_top5=shap_day_top5,
        shap_night_top5=shap_night_top5,
        distribution_shift_score=round(dist_score, 3),
        distribution_shift_label=dist_label,
        pue_was_imputed=pue_was_imputed,
        footprint_was_estimated=footprint_was_estimated,
        gee_extraction_status=gee_status,
        cool_island_risk=cool_island_risk,
        day_prediction_method=day_method,
        model_version=models["eval_report"].get("version", "unknown"),
        prediction_id=prediction_id,
    )


# ── Climate-stratified GEE fallback defaults ─────────────────────────────────
_ERA5_DEFAULTS_BY_CLIMATE = {
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


def _get_climate_stratified_defaults(climate_zone: str) -> dict:
    defaults = _ERA5_DEFAULTS_BY_CLIMATE.get(
        climate_zone, _ERA5_DEFAULTS_BY_CLIMATE["Cfa"]
    )
    return {**defaults, "gee_status": "climate_fallback"}
