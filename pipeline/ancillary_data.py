"""
Stage 2b: Ancillary Data Extraction

Extracts non-LST spatial covariates from GEE for all sites and ring zones:
  - NDVI: Annual max from Landsat growing-season composite (Apr-Oct)
  - Land cover fractions: ESA WorldCover 2021 (built-up, tree, bare, water, cropland)
  - Population density: WorldPop 100m
  - Elevation: SRTM 30m
  - Snow cover days: MODIS MOD10A2 (Canadian sites only)

These covariates support CEM (Coarsened Exact Matching) for control selection
and serve as model inputs for the DCTII composite index.

Uses same site geometry and ring zone system as lst_ingestion.py.
"""

import os
import logging
from datetime import datetime, date
from typing import Optional

import ee
from google.cloud import bigquery

logger = logging.getLogger("dctii.ancillary_data")

# ---------------------------------------------------------------------------
# Constants (shared with lst_ingestion.py)
# ---------------------------------------------------------------------------
GCP_PROJECT = os.environ.get("GCP_PROJECT", "oil-tank-monitoring-123")
BQ_STAGING_DATASET = "dctii_staging"
BQ_REF_DATASET = "dctii_ref"

RING_ZONES = [
    ("footprint", 0, 1),
    ("near", 1, 3),
    ("buffer_1", 3, 5),
    ("buffer_2", 5, 10),
    ("control_near", 10, 20),
    ("control_far", 20, 35),
]

# GEE collection IDs
LANDSAT8_COLLECTION = "LANDSAT/LC08/C02/T1_L2"
LANDSAT9_COLLECTION = "LANDSAT/LC09/C02/T1_L2"
WORLDCOVER_COLLECTION = "ESA/WorldCover/v200"
WORLDPOP_COLLECTION = "WorldPop/GP/100m/pop"
SRTM_COLLECTION = "USGS/SRTMGL1_003"
MODIS_SNOW_COLLECTION = "MODIS/061/MOD10A2"

# ESA WorldCover class values (2021)
# 10=Tree, 20=Shrub, 30=Grassland, 40=Cropland, 50=Built-up,
# 60=Bare, 70=Snow/Ice, 80=Water, 90=Herbaceous wetland, 95=Mangroves, 100=Moss
WORLDCOVER_CLASSES = {
    "tree_cover": 10,
    "shrubland": 20,
    "grassland": 30,
    "cropland": 40,
    "built_up": 50,
    "bare": 60,
    "snow_ice": 70,
    "water": 80,
    "wetland": 90,
}

# Canadian regions for snow cover extraction
CANADIAN_REGIONS = {"TOR", "MTL"}

# Growing season for NDVI (April-October, inclusive)
GROWING_SEASON_START_MONTH = 4
GROWING_SEASON_END_MONTH = 10

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_ee_initialized = False
_bq_client = None


def _init_gee():
    global _ee_initialized
    if not _ee_initialized:
        ee.Initialize(project=GCP_PROJECT)
        _ee_initialized = True


def _get_bq_client():
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=GCP_PROJECT)
    return _bq_client


def _make_ring_geometry(lat: float, lon: float, inner_km: float, outer_km: float):
    """Create annular ring geometry (donut) centered on site."""
    point = ee.Geometry.Point([lon, lat])
    outer = point.buffer(outer_km * 1000)
    if inner_km > 0:
        inner = point.buffer(inner_km * 1000)
        return outer.difference(inner)
    return outer


# ---------------------------------------------------------------------------
# NDVI extraction (Landsat growing-season max composite)
# ---------------------------------------------------------------------------

def _compute_ndvi(image):
    """Compute NDVI from Landsat C2L2 SR bands."""
    nir = image.select("SR_B5").multiply(0.0000275).add(-0.2)
    red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    return nir.subtract(red).divide(nir.add(red)).rename("NDVI")


def _mask_landsat_clouds(image):
    """Cloud mask for Landsat using QA_PIXEL band."""
    qa = image.select("QA_PIXEL")
    cloud_mask = (
        qa.bitwiseAnd(1 << 1).eq(0)   # dilated cloud
        .And(qa.bitwiseAnd(1 << 2).eq(0))  # cirrus
        .And(qa.bitwiseAnd(1 << 3).eq(0))  # cloud
        .And(qa.bitwiseAnd(1 << 4).eq(0))  # shadow
    )
    return image.updateMask(cloud_mask)


def _extract_ndvi_for_site(site: dict, year: int) -> list:
    """Extract annual growing-season max NDVI for each ring zone.
    
    Uses batched GEE computation — all zones reduced in a single getInfo() call
    to minimize API overhead. Scale adapts to zone size (30m for footprint/near,
    100m for larger zones) to keep computation feasible.
    """
    lat, lon = site["latitude"], site["longitude"]

    # Growing season: April 1 to October 31
    start_date = f"{year}-{GROWING_SEASON_START_MONTH:02d}-01"
    end_date = f"{year}-{GROWING_SEASON_END_MONTH + 1:02d}-01"

    # Combine Landsat 8 + 9
    l8 = (ee.ImageCollection(LANDSAT8_COLLECTION)
          .filterDate(start_date, end_date)
          .filterBounds(ee.Geometry.Point([lon, lat]).buffer(35000)))
    l9 = (ee.ImageCollection(LANDSAT9_COLLECTION)
          .filterDate(start_date, end_date)
          .filterBounds(ee.Geometry.Point([lon, lat]).buffer(35000)))

    combined = l8.merge(l9)
    ndvi_collection = combined.map(_mask_landsat_clouds).map(_compute_ndvi)

    ndvi_max = ndvi_collection.max()
    ndvi_mean = ndvi_collection.mean()

    # Batch all zones into a single dictionary for one getInfo() call
    batch = {}
    for zone_name, inner_km, outer_km in RING_ZONES:
        geom = _make_ring_geometry(lat, lon, inner_km, outer_km)
        # Adaptive scale: finer for small zones, coarser for large ones
        scale = 30 if outer_km <= 5 else 100

        batch[f"{zone_name}_max"] = ndvi_max.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e8,
        ).get("NDVI")
        batch[f"{zone_name}_mean"] = ndvi_mean.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e8,
        ).get("NDVI")

    # Single API call for all zones
    all_stats = ee.Dictionary(batch).getInfo()

    results = []
    for zone_name, _, _ in RING_ZONES:
        results.append({
            "zone_name": zone_name,
            "ndvi_max": all_stats.get(f"{zone_name}_max"),
            "ndvi_mean": all_stats.get(f"{zone_name}_mean"),
        })

    return results


# ---------------------------------------------------------------------------
# Land cover fractions (ESA WorldCover 2021)
# ---------------------------------------------------------------------------

def _extract_landcover_for_site(site: dict) -> list:
    """Extract land cover class fractions for each ring zone.
    
    Splits into 2 batches (3 zones each) to avoid GEE aggregation limits.
    Only extracts the 5 key classes needed for CEM matching.
    """
    lat, lon = site["latitude"], site["longitude"]

    worldcover = ee.ImageCollection(WORLDCOVER_COLLECTION).first().select("Map")

    # Key classes for CEM matching
    key_classes = {
        "built_up": 50,
        "tree_cover": 10,
        "bare": 60,
        "water": 80,
        "cropland": 40,
    }

    # Process in 2 batches to avoid aggregation limits
    all_results = {}
    zone_batches = [RING_ZONES[:3], RING_ZONES[3:]]

    for zone_batch in zone_batches:
        batch = {}
        for zone_name, inner_km, outer_km in zone_batch:
            geom = _make_ring_geometry(lat, lon, inner_km, outer_km)
            scale = 10 if outer_km <= 5 else 30

            total = worldcover.reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geom,
                scale=scale,
                maxPixels=1e9,
            ).get("Map")
            batch[f"{zone_name}_total"] = total

            for class_name, class_value in key_classes.items():
                class_count = worldcover.eq(class_value).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=geom,
                    scale=scale,
                    maxPixels=1e9,
                ).get("Map")
                batch[f"{zone_name}_{class_name}"] = ee.Number(class_count).divide(ee.Number(total))

        stats = ee.Dictionary(batch).getInfo()
        all_results.update(stats)

    results = []
    for zone_name, _, _ in RING_ZONES:
        results.append({
            "zone_name": zone_name,
            "impervious_fraction": all_results.get(f"{zone_name}_built_up", 0),
            "tree_cover_fraction": all_results.get(f"{zone_name}_tree_cover", 0),
            "bare_fraction": all_results.get(f"{zone_name}_bare", 0),
            "water_fraction": all_results.get(f"{zone_name}_water", 0),
            "cropland_fraction": all_results.get(f"{zone_name}_cropland", 0),
        })

    return results


# ---------------------------------------------------------------------------
# Population density (WorldPop 100m)
# ---------------------------------------------------------------------------

def _extract_population_for_site(site: dict, year: int) -> list:
    """Extract population density and total for each ring zone.
    
    WorldPop stores per-country images. We mosaic all images covering the site
    area to handle cross-border sites (e.g., near US-Canada border).
    Uses latest available year if requested year unavailable (max 2020 on GEE).
    """
    lat, lon = site["latitude"], site["longitude"]
    point = ee.Geometry.Point([lon, lat])

    # WorldPop on GEE covers 2000-2020
    pop_year = min(year, 2020)
    pop_col = (ee.ImageCollection(WORLDPOP_COLLECTION)
               .filterDate(f"{pop_year}-01-01", f"{pop_year}-12-31")
               .filterBounds(point.buffer(35000))
               .select("population"))

    # Fallback to latest available year
    pop_count = pop_col.size().getInfo()
    if pop_count == 0:
        for fallback_year in range(pop_year - 1, 1999, -1):
            pop_col = (ee.ImageCollection(WORLDPOP_COLLECTION)
                       .filterDate(f"{fallback_year}-01-01", f"{fallback_year}-12-31")
                       .filterBounds(point.buffer(35000))
                       .select("population"))
            if pop_col.size().getInfo() > 0:
                logger.info(f"  WorldPop: using {fallback_year} (requested {year})")
                break
        else:
            logger.warning(f"  No WorldPop data available")
            return [{"zone_name": z[0], "population_density": None, "population_total": None}
                    for z in RING_ZONES]

    # Mosaic all country images covering the area
    pop = pop_col.mosaic()

    batch = {}
    for zone_name, inner_km, outer_km in RING_ZONES:
        geom = _make_ring_geometry(lat, lon, inner_km, outer_km)
        scale = 100

        stats = pop.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.sum(), sharedInputs=True
            ),
            geometry=geom,
            scale=scale,
            maxPixels=1e9,
        )

        # Convert mean (persons/pixel at 100m) to density (persons/km2)
        # Each 100m pixel = 0.01 km2, so density = mean * 100
        mean_val = stats.get("population_mean")
        batch[f"{zone_name}_density"] = ee.Algorithms.If(
            ee.Algorithms.IsEqual(mean_val, None), None,
            ee.Number(mean_val).multiply(100)
        )
        batch[f"{zone_name}_total"] = stats.get("population_sum")

    all_stats = ee.Dictionary(batch).getInfo()

    results = []
    for zone_name, _, _ in RING_ZONES:
        results.append({
            "zone_name": zone_name,
            "population_density": all_stats.get(f"{zone_name}_density"),
            "population_total": all_stats.get(f"{zone_name}_total"),
        })

    return results


# ---------------------------------------------------------------------------
# Elevation (SRTM 30m)
# ---------------------------------------------------------------------------

def _extract_elevation_for_site(site: dict) -> list:
    """Extract elevation mean and std for each ring zone."""
    lat, lon = site["latitude"], site["longitude"]

    srtm = ee.Image(SRTM_COLLECTION).select("elevation")

    results = []
    for zone_name, inner_km, outer_km in RING_ZONES:
        geom = _make_ring_geometry(lat, lon, inner_km, outer_km)
        scale = 30  # SRTM native resolution

        stats = srtm.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                ee.Reducer.stdDev(), sharedInputs=True
            ),
            geometry=geom,
            scale=scale,
            maxPixels=1e9,
        )

        results.append({
            "zone_name": zone_name,
            "elevation_mean_m": stats.get("elevation_mean").getInfo(),
            "elevation_std_m": stats.get("elevation_stdDev").getInfo(),
        })

    return results


# ---------------------------------------------------------------------------
# Snow cover days (MODIS MOD10A2 — Canadian sites only)
# ---------------------------------------------------------------------------

def _extract_snow_cover_for_site(site: dict, year: int) -> list:
    """Extract annual snow-covered days for each ring zone (Canada only).
    
    Batches all zone computations into a single getInfo() call.
    """
    lat, lon = site["latitude"], site["longitude"]

    snow_col = (ee.ImageCollection(MODIS_SNOW_COLLECTION)
                .filterDate(f"{year}-01-01", f"{year+1}-01-01")
                .filterBounds(ee.Geometry.Point([lon, lat]).buffer(35000))
                .select("Maximum_Snow_Extent"))

    def _to_snow_binary(img):
        return img.eq(200).rename("snow")

    snow_binary = snow_col.map(_to_snow_binary)
    snow_days_img = snow_binary.sum().multiply(8)

    batch = {}
    for zone_name, inner_km, outer_km in RING_ZONES:
        geom = _make_ring_geometry(lat, lon, inner_km, outer_km)
        scale = 500

        batch[zone_name] = snow_days_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=scale,
            maxPixels=1e8,
        ).get("snow")

    all_stats = ee.Dictionary(batch).getInfo()

    results = []
    for zone_name, _, _ in RING_ZONES:
        results.append({
            "zone_name": zone_name,
            "snow_cover_days": all_stats.get(zone_name),
        })

    return results


# ---------------------------------------------------------------------------
# BQ writer
# ---------------------------------------------------------------------------

def write_covariates_to_bq(rows: list):
    """Write covariate rows to BigQuery site_covariates table."""
    if not rows:
        return 0
    client = _get_bq_client()
    table_id = f"{GCP_PROJECT}.{BQ_STAGING_DATASET}.site_covariates"
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        logger.error(f"BQ insert errors: {errors[:3]}")
        raise RuntimeError(f"BigQuery insert failed: {len(errors)} errors")
    logger.info(f"Inserted {len(rows)} covariate rows into {table_id}")
    return len(rows)


# ---------------------------------------------------------------------------
# Site loader
# ---------------------------------------------------------------------------

def load_sites(region_code: str = None) -> list:
    """Load site registry from BigQuery."""
    client = _get_bq_client()
    query = f"SELECT * FROM `{GCP_PROJECT}.{BQ_REF_DATASET}.site_registry`"
    if region_code:
        query += f" WHERE region_code = '{region_code}'"
    return [dict(row) for row in client.query(query).result()]


# ---------------------------------------------------------------------------
# Orchestrator: extract all covariates for a region + year
# ---------------------------------------------------------------------------

def extract_covariates_region(
    region_code: str,
    year: int,
) -> dict:
    """
    Extract all ancillary covariates for a region and year.

    Returns dict with counts per data layer.
    """
    _init_gee()
    sites = load_sites(region_code)
    logger.info(f"Extracting covariates for {region_code} {year}, {len(sites)} sites")

    is_canadian = region_code in CANADIAN_REGIONS
    all_rows = []

    for site in sites:
        site_id = site["site_id"]
        logger.info(f"  Processing {site_id}...")

        # Extract each data layer
        try:
            ndvi_data = _extract_ndvi_for_site(site, year)
        except Exception as e:
            logger.warning(f"  NDVI extraction failed for {site_id}: {e}")
            ndvi_data = [{"zone_name": z[0], "ndvi_max": None, "ndvi_mean": None} for z in RING_ZONES]

        try:
            lc_data = _extract_landcover_for_site(site)
        except Exception as e:
            logger.warning(f"  Land cover extraction failed for {site_id}: {e}")
            lc_data = [{"zone_name": z[0], "impervious_fraction": None, "tree_cover_fraction": None,
                        "bare_fraction": None, "water_fraction": None, "cropland_fraction": None} for z in RING_ZONES]

        try:
            pop_data = _extract_population_for_site(site, year)
        except Exception as e:
            logger.warning(f"  Population extraction failed for {site_id}: {e}")
            pop_data = [{"zone_name": z[0], "population_density": None, "population_total": None} for z in RING_ZONES]

        try:
            elev_data = _extract_elevation_for_site(site)
        except Exception as e:
            logger.warning(f"  Elevation extraction failed for {site_id}: {e}")
            elev_data = [{"zone_name": z[0], "elevation_mean_m": None, "elevation_std_m": None} for z in RING_ZONES]

        snow_data = None
        if is_canadian:
            try:
                snow_data = _extract_snow_cover_for_site(site, year)
            except Exception as e:
                logger.warning(f"  Snow cover extraction failed for {site_id}: {e}")
                snow_data = [{"zone_name": z[0], "snow_cover_days": None} for z in RING_ZONES]

        # Merge all data layers into BQ rows
        now = datetime.utcnow().isoformat() + "Z"
        for i, (zone_name, _, _) in enumerate(RING_ZONES):
            sources = ["Landsat_NDVI", "ESA_WorldCover_v200", "WorldPop_100m", "SRTM_30m"]
            if is_canadian:
                sources.append("MODIS_MOD10A2")

            row = {
                "site_id": site_id,
                "region_code": region_code,
                "zone_name": zone_name,
                "covariate_date": f"{year}-01-01",
                "ndvi_max": ndvi_data[i].get("ndvi_max"),
                "ndvi_mean": ndvi_data[i].get("ndvi_mean"),
                "impervious_fraction": lc_data[i].get("impervious_fraction"),
                "tree_cover_fraction": lc_data[i].get("tree_cover_fraction"),
                "bare_fraction": lc_data[i].get("bare_fraction"),
                "water_fraction": lc_data[i].get("water_fraction"),
                "cropland_fraction": lc_data[i].get("cropland_fraction"),
                "population_density": pop_data[i].get("population_density"),
                "population_total": pop_data[i].get("population_total"),
                "elevation_mean_m": elev_data[i].get("elevation_mean_m"),
                "elevation_std_m": elev_data[i].get("elevation_std_m"),
                "snow_cover_days": snow_data[i].get("snow_cover_days") if snow_data else None,
                "valid_pixel_fraction": None,  # Computed per-layer if needed
                "source_products": ",".join(sources),
                "created_ts": now,
            }
            all_rows.append(row)

    # Write to BQ
    n_written = write_covariates_to_bq(all_rows)
    logger.info(f"Completed {region_code} {year}: {n_written} covariate rows")
    return {"rows": n_written, "sites": len(sites)}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Extract ancillary covariates")
    parser.add_argument("--region", type=str, required=True,
                        help="Region code (PHX, HOU, NOVA, CTX, TOR, MTL)")
    parser.add_argument("--year", type=int, required=True,
                        help="Year for temporal covariates (NDVI, population)")
    args = parser.parse_args()

    result = extract_covariates_region(args.region, args.year)
    print(f"Done: {result}")
