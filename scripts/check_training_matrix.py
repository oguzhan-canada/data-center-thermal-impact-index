"""Quick health check on the DCTII-Predict training matrix."""
from google.cloud import bigquery

PROJECT = "oil-tank-monitoring-123"
client = bigquery.Client(project=PROJECT)

sql = f"""
WITH indicators AS (
    SELECT
        i.year,
        i.site_id,
        i.delta_t_day         AS label_delta_t_day,
        i.delta_t_night       AS label_delta_t_night,
        i.heat_island_area_km2,
        i.population_exposed,
        i.waste_heat_flux_wm2,
        i.min_monthly_reliability AS estimation_method,
        i.fraction_reliable_months
    FROM `{PROJECT}.dctii_curated.site_indicators` i
    WHERE i.delta_t_night IS NOT NULL
      AND i.delta_t_day   IS NOT NULL
      AND i.delta_t_night BETWEEN -1.0 AND 8.0
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
    FROM `{PROJECT}.dctii_serving.did_results`
    WHERE data_version = (SELECT MAX(data_version) FROM `{PROJECT}.dctii_serving.did_results`)
    GROUP BY site_id, cohort_year
),
scores AS (
    SELECT site_id, year, dctii_score, weighting_scheme
    FROM `{PROJECT}.dctii_serving.dctii_scores`
    WHERE weighting_scheme = 'expert'
      AND data_version = (SELECT MAX(data_version) FROM `{PROJECT}.dctii_serving.dctii_scores`)
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
    FROM `{PROJECT}.dctii_ref.site_registry`
),
covariates AS (
    SELECT
        site_id,
        EXTRACT(YEAR FROM covariate_date) AS covariate_year,
        ndvi_max                          AS ndvi_growing_max,
        impervious_fraction,
        tree_cover_fraction,
        bare_fraction,
        population_density,
        elevation_mean_m                  AS elevation_m,
        snow_cover_days,
        FALSE                             AS covariate_year_proxy
    FROM `{PROJECT}.dctii_staging.site_covariates`
    WHERE zone_name = 'footprint'
),
training_matrix AS (
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
)
SELECT
    COUNT(*)                                          AS total_rows,
    COUNT(CASE WHEN covariate_year_proxy THEN 1 END)  AS proxy_rows,
    COUNT(CASE WHEN att_night IS NOT NULL THEN 1 END) AS causal_label_rows,
    AVG(fraction_reliable_months)                     AS avg_reliability,
    MIN(final_label_delta_t_night)                    AS min_dt_night,
    MAX(final_label_delta_t_night)                    AS max_dt_night,
    AVG(final_label_delta_t_night)                    AS avg_dt_night,
    STDDEV(final_label_delta_t_night)                 AS std_dt_night
FROM training_matrix
"""

# --- Diagnostics first ---
print("=== Table diagnostics ===")
diag_queries = {
    "site_indicators": f"SELECT COUNT(*) as n, COUNT(DISTINCT site_id) as sites FROM `{PROJECT}.dctii_curated.site_indicators`",
    "site_registry": f"SELECT COUNT(*) as n, COUNT(DISTINCT site_id) as sites FROM `{PROJECT}.dctii_ref.site_registry`",
    "did_results": f"SELECT COUNT(*) as n, COUNT(DISTINCT site_id) as sites FROM `{PROJECT}.dctii_serving.did_results`",
    "dctii_scores": f"SELECT COUNT(*) as n, COUNT(DISTINCT site_id) as sites FROM `{PROJECT}.dctii_serving.dctii_scores`",
    "site_covariates": f"SELECT COUNT(*) as n, COUNT(DISTINCT site_id) as sites FROM `{PROJECT}.dctii_staging.site_covariates` WHERE zone_name='footprint'",
}
for name, q in diag_queries.items():
    r = list(client.query(q).result())[0]
    print(f"  {name}: {r.n} rows, {r.sites} sites")

# Check join key overlap
print("\n=== Join key overlap ===")
overlap = f"""
SELECT
  (SELECT COUNT(DISTINCT site_id) FROM `{PROJECT}.dctii_curated.site_indicators`) AS indicator_sites,
  (SELECT COUNT(DISTINCT site_id) FROM `{PROJECT}.dctii_ref.site_registry`) AS registry_sites,
  (SELECT COUNT(DISTINCT i.site_id) FROM `{PROJECT}.dctii_curated.site_indicators` i
   JOIN `{PROJECT}.dctii_ref.site_registry` r USING (site_id)) AS joined_sites
"""
r = list(client.query(overlap).result())[0]
print(f"  indicator sites: {r.indicator_sites}")
print(f"  registry sites:  {r.registry_sites}")
print(f"  joined sites:    {r.joined_sites}")

# Sample indicator site_ids vs registry site_ids
print("\n=== Sample site_id formats ===")
q1 = f"SELECT DISTINCT site_id FROM `{PROJECT}.dctii_curated.site_indicators` LIMIT 5"
q2 = f"SELECT DISTINCT site_id FROM `{PROJECT}.dctii_ref.site_registry` LIMIT 5"
print("  indicators:", [r.site_id for r in client.query(q1).result()])
print("  registry:  ", [r.site_id for r in client.query(q2).result()])

# Check reliability filter
print("\n=== Reliability filter ===")
q3 = f"""
SELECT
  COUNT(*) AS total,
  COUNT(CASE WHEN fraction_reliable_months >= 0.5 THEN 1 END) AS pass_frac,
  COUNT(CASE WHEN min_monthly_reliability != 'UNRELIABLE' THEN 1 END) AS pass_rel
FROM `{PROJECT}.dctii_curated.site_indicators`
"""
r = list(client.query(q3).result())[0]
print(f"  total: {r.total}, pass_frac>=0.5: {r.pass_frac}, pass_rel!=UNRELIABLE: {r.pass_rel}")

# Now run the main health check
print("\n=== Training Matrix Health Check ===")
result = list(client.query(sql).result())
r = result[0]
print(f"Total rows:         {r.total_rows}")
print(f"Proxy rows:         {r.proxy_rows}")
print(f"Causal label rows:  {r.causal_label_rows}")
if r.avg_reliability is not None:
    print(f"Avg reliability:    {r.avg_reliability:.4f}")
    print(f"Min DT night:       {r.min_dt_night:.4f}")
    print(f"Max DT night:       {r.max_dt_night:.4f}")
    print(f"Avg DT night:       {r.avg_dt_night:.4f}")
    print(f"Std DT night:       {r.std_dt_night:.4f}")
else:
    print("WARNING: No rows passed filters — all metrics are NULL")
