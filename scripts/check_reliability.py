"""Check fraction_reliable_months distribution."""
from google.cloud import bigquery

PROJECT = "oil-tank-monitoring-123"
client = bigquery.Client(project=PROJECT)

q1 = f"""
SELECT
  MIN(fraction_reliable_months) as min_frm,
  MAX(fraction_reliable_months) as max_frm,
  AVG(fraction_reliable_months) as avg_frm,
  COUNTIF(fraction_reliable_months >= 0.5) as ge_50pct,
  COUNTIF(fraction_reliable_months >= 0.05) as ge_5pct,
  COUNTIF(fraction_reliable_months IS NULL) as null_count,
  COUNT(*) as total
FROM `{PROJECT}.dctii_curated.site_indicators`
"""
r = list(client.query(q1).result())[0]
print(f"min: {r.min_frm}, max: {r.max_frm}, avg: {r.avg_frm}")
print(f">=0.5: {r.ge_50pct}, >=0.05: {r.ge_5pct}, null: {r.null_count}, total: {r.total}")

q2 = f"""
SELECT site_id, year, fraction_reliable_months, min_monthly_reliability
FROM `{PROJECT}.dctii_curated.site_indicators`
ORDER BY fraction_reliable_months DESC
LIMIT 10
"""
print("\nTop 10 by fraction_reliable_months:")
for r in client.query(q2).result():
    print(f"  {r.site_id} {r.year}: frac={r.fraction_reliable_months}, min_rel={r.min_monthly_reliability}")

# Also check did_results
q3 = f"SELECT COUNT(*) as n FROM `{PROJECT}.dctii_serving.did_results`"
r = list(client.query(q3).result())[0]
print(f"\ndid_results rows: {r.n}")
