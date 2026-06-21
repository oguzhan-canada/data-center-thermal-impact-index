"""Investigate PHX_002 and PHX_006 outlier sites."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.cloud import bigquery
import pandas as pd

BQ = bigquery.Client(project="oil-tank-monitoring-123")

PHX_SQL = """
WITH latest_cov AS (
    SELECT site_id, ndvi_max AS ndvi_growing_max, impervious_fraction,
           population_density, elevation_mean_m AS elevation_m,
           ROW_NUMBER() OVER (PARTITION BY site_id ORDER BY covariate_date DESC) AS rn
    FROM `oil-tank-monitoring-123.dctii_staging.site_covariates`
    WHERE zone_name = 'footprint'
),
latest_ind AS (
    SELECT site_id, year, delta_t_night, delta_t_day,
           waste_heat_flux_wm2, heat_island_area_km2,
           ROW_NUMBER() OVER (PARTITION BY site_id ORDER BY year DESC) AS rn
    FROM `oil-tank-monitoring-123.dctii_curated.site_indicators`
    WHERE delta_t_night IS NOT NULL
),
night_std AS (
    SELECT site_id, STDDEV(delta_t_night) AS dt_night_stddev
    FROM `oil-tank-monitoring-123.dctii_curated.site_indicators`
    GROUP BY site_id
),
all_years AS (
    SELECT site_id, year, delta_t_night, delta_t_day
    FROM `oil-tank-monitoring-123.dctii_curated.site_indicators`
    WHERE site_id IN ('PHX_002', 'PHX_006')
    ORDER BY site_id, year
)
SELECT
    s.site_id, s.capacity_mw, s.pue_estimate, s.pue_source,
    s.load_factor, s.cooling_type, s.footprint_km2,
    s.confidence_tier, s.activation_year, s.cluster_id,
    i.delta_t_night AS actual_dt_night,
    i.delta_t_day AS actual_dt_day,
    i.waste_heat_flux_wm2, i.heat_island_area_km2,
    c.ndvi_growing_max, c.impervious_fraction,
    c.population_density, c.elevation_m,
    n.dt_night_stddev
FROM `oil-tank-monitoring-123.dctii_ref.site_registry` s
LEFT JOIN latest_ind i ON i.site_id = s.site_id AND i.rn = 1
LEFT JOIN latest_cov c ON c.site_id = s.site_id AND c.rn = 1
LEFT JOIN night_std n ON n.site_id = s.site_id
WHERE s.site_id IN ('PHX_002', 'PHX_006')
"""

print("=== PHX Outlier Site Details ===")
df = BQ.query(PHX_SQL).to_dataframe()
print(df.T.to_string())

# Multi-year trend
TREND_SQL = """
SELECT site_id, year, delta_t_night, delta_t_day
FROM `oil-tank-monitoring-123.dctii_curated.site_indicators`
WHERE site_id IN ('PHX_002', 'PHX_006')
ORDER BY site_id, year
"""
print("\n=== Multi-year trend ===")
trend = BQ.query(TREND_SQL).to_dataframe()
print(trend.to_string(index=False))

# Compare to other PHX sites
PHX_ALL_SQL = """
WITH latest_ind AS (
    SELECT site_id, delta_t_night, delta_t_day,
           ROW_NUMBER() OVER (PARTITION BY site_id ORDER BY year DESC) AS rn
    FROM `oil-tank-monitoring-123.dctii_curated.site_indicators`
    WHERE delta_t_night IS NOT NULL
)
SELECT s.site_id, s.capacity_mw, s.pue_estimate, s.cooling_type,
       s.footprint_km2, s.confidence_tier,
       i.delta_t_night, i.delta_t_day
FROM `oil-tank-monitoring-123.dctii_ref.site_registry` s
LEFT JOIN latest_ind i ON i.site_id = s.site_id AND i.rn = 1
WHERE s.site_id LIKE 'PHX_%'
ORDER BY s.site_id
"""
print("\n=== All PHX sites comparison ===")
phx_all = BQ.query(PHX_ALL_SQL).to_dataframe()
print(phx_all.to_string(index=False))
