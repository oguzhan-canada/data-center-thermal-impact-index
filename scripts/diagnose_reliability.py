"""Diagnose reliability data in site_indicators."""
from google.cloud import bigquery

PROJECT = "oil-tank-monitoring-123"
client = bigquery.Client(project=PROJECT)

# 1. Reliability breakdown
print("=== Reliability breakdown ===")
q1 = f"""
SELECT
  min_monthly_reliability,
  COUNT(*) AS row_count,
  AVG(fraction_reliable_months) AS avg_fraction,
  MIN(delta_t_night) AS min_dt_night,
  MAX(delta_t_night) AS max_dt_night
FROM `{PROJECT}.dctii_curated.site_indicators`
GROUP BY min_monthly_reliability
ORDER BY row_count DESC
"""
for r in client.query(q1).result():
    avg_f = f"{r.avg_fraction:.4f}" if r.avg_fraction is not None else "NULL"
    min_n = f"{r.min_dt_night:.4f}" if r.min_dt_night is not None else "NULL"
    max_n = f"{r.max_dt_night:.4f}" if r.max_dt_night is not None else "NULL"
    print(f"  {r.min_monthly_reliability:25s}  count={r.row_count:4d}  avg_frac={avg_f}  dt_night=[{min_n}, {max_n}]")

# 2. Full schema of site_indicators
print("\n=== site_indicators schema ===")
t = client.get_table(f"{PROJECT}.dctii_curated.site_indicators")
for f in t.schema:
    print(f"  {f.name} ({f.field_type})")

# 3. Sample row
print("\n=== Sample row ===")
q2 = f"SELECT * FROM `{PROJECT}.dctii_curated.site_indicators` LIMIT 1"
for r in client.query(q2).result():
    for k, v in dict(r).items():
        print(f"  {k}: {v}")

# 4. Check non-null delta_t columns
print("\n=== Label availability ===")
q3 = f"""
SELECT
  COUNT(*) AS total,
  COUNTIF(delta_t_day IS NOT NULL) AS has_dt_day,
  COUNTIF(delta_t_night IS NOT NULL) AS has_dt_night,
  COUNTIF(delta_t_day IS NOT NULL AND delta_t_night IS NOT NULL) AS has_both,
  AVG(delta_t_day) AS avg_dt_day,
  AVG(delta_t_night) AS avg_dt_night,
  STDDEV(delta_t_day) AS std_dt_day,
  STDDEV(delta_t_night) AS std_dt_night
FROM `{PROJECT}.dctii_curated.site_indicators`
"""
r = list(client.query(q3).result())[0]
print(f"  total: {r.total}")
print(f"  has_dt_day: {r.has_dt_day}, has_dt_night: {r.has_dt_night}, has_both: {r.has_both}")
if r.avg_dt_day is not None:
    print(f"  avg_dt_day: {r.avg_dt_day:.4f} +/- {r.std_dt_day:.4f}")
    print(f"  avg_dt_night: {r.avg_dt_night:.4f} +/- {r.std_dt_night:.4f}")
