# DCTII — Data Center Thermal Impact Index

**A satellite-grounded platform for measuring and predicting the urban-heat-island impact of data centers.**

DCTII quantifies how much a data center warms its surroundings using a decade of public satellite land-surface-temperature (LST) data, control-matched comparison zones, and a physically-anchored 0–100 score. It ships with a **predictive ML module (DCTII-Predict)** that estimates the thermal footprint of a *planned* facility — before any satellite data for that site exists.

> Built on a 10-year panel of **42 data-center sites across 6 metro regions** (Phoenix, Houston, Central Texas, Northern Virginia, Toronto, Montreal), with a FastAPI service, an interactive Leaflet/Chart.js dashboard, and reproducible Terraform infrastructure on Google Cloud.

---

## What it does

1. **Retrospective scoring (DCTII).** Ingests MODIS Terra/Aqua + Landsat 8/9 LST, extracts concentric ring zones around each site, isolates the facility's thermal signal via Coarsened Exact Matching (CEM) control sites, and composes a 0–100 impact score driven by day/night ΔT, waste-heat flux, heat-island area, and population exposure.
2. **Prospective prediction (DCTII-Predict).** Given a candidate location and proposed specs (capacity MW, cooling type, PUE, footprint), predicts day/night ΔT with conformal prediction intervals, derives secondary indicators, composes the DCTII score, and returns SHAP explanations plus a distribution-shift warning.

---

## Architecture

```
┌────────────────────────────┬───────────────────────────────────────────┐
│     OFFLINE: TRAINING      │            ONLINE: INFERENCE                │
│                            │                                            │
│  BigQuery:                 │   POST /api/v1/predict                     │
│   site_indicators          │   { lat, lon, capacity_mw, pue,            │
│   site_covariates          │     cooling_type, footprint_km2, ... }     │
│   site_registry            │            │                               │
│        │                   │            ▼                               │
│        ▼                   │   1. Geocode → lat/lon                     │
│  pipeline/predict_train.py │   2. GEE feature extraction                │
│   build_training_matrix    │   3. engineer_features()                   │
│   engineer_features()      │   4. Load model from GCS                   │
│   tune (Optuna, LORO CV)   │   5. lgbm.predict() → ΔT_day, ΔT_night     │
│   train_final_model()      │   6. derive secondary indicators           │
│   calibrate (cross-conf.)  │   7. compose DCTII score                   │
│   compute_shap()           │   8. conformal prediction interval         │
│        │                   │   9. distribution-shift score              │
│        ▼                   │  10. write to BQ predictions table         │
│  GCS: dctii-model-*/       │  11. return full JSON response             │
│       predict/v{N}/        │                                            │
└────────────────────────────┴───────────────────────────────────────────┘
```

---

## Repository structure

```
data-center-thermal-impact-index/
├── api/
│   └── dctii_api.py            FastAPI service (Cloud Run): scoring + POST /predict
├── pipeline/
│   ├── predict_train.py        ML training orchestrator (LightGBM, Optuna, conformal)
│   ├── predict_infer.py        Inference utilities (feature extraction, scoring)
│   ├── run_predict_train.py    CLI entrypoint for training
│   ├── ancillary_data.py       Earth Engine covariate extraction
│   ├── indicator_compute.py    Waste-heat-flux & indicator formulas
│   ├── dctii_calculator.py     Normalization & 0–100 score composition
│   ├── sensitivity.py          Spearman-ρ sensitivity sweeps
│   ├── validation.py           Data-quality validation
│   ├── populate_serving.py     BigQuery serving-layer orchestrator
│   └── pipeline_run.py         Pipeline-run lineage tracking
├── scripts/                    Diagnostics & one-off analysis utilities
│   └── download_koppen.py      Fetches the Köppen-Geiger climate raster (not vendored)
├── tests/                      Unit, integration & golden-regression tests (pytest)
├── dashboard/
│   └── dctii.html              Interactive Leaflet map + Chart.js + Site Planner
├── terraform/
│   ├── main.tf                 GCP infra (BigQuery, Cloud Run, GCS, monitoring)
│   └── .terraform.lock.hcl     Pinned provider versions
├── docs/
│   └── DCTII_Predict_Project.md Full technical design & methodology
├── requirements-api.txt        Lightweight API dependencies
├── requirements-pipeline.txt   Heavy pipeline/ML dependencies
├── Dockerfile                  API container
├── Dockerfile.pipeline         Pipeline container
├── .env.example                Configuration template
└── LICENSE                     MIT
```

---

## Tech stack

| Layer | Tooling |
|-------|---------|
| Remote sensing | Google Earth Engine (MODIS, Landsat 8/9, NDVI, Köppen-Geiger) |
| Data warehouse | BigQuery (`dctii_*` datasets) |
| ML | LightGBM (quantile), Optuna (TPE), SHAP, cross-conformal prediction, scikit-learn |
| Causal / stats | CEM matching, statsmodels (Conley spatial HAC) |
| Serving | FastAPI + Uvicorn on Cloud Run |
| Frontend | Leaflet, Chart.js (single-file dashboard) |
| Infra | Terraform (Google provider), Cloud Build, Docker |

---

## Getting started

### Prerequisites
- Python 3.11+
- A Google Cloud project with BigQuery, Cloud Storage, and Earth Engine enabled
- `gcloud` authenticated (`gcloud auth application-default login`)

### Install

```bash
# API only (lightweight)
pip install -r requirements-api.txt

# Full pipeline + ML
pip install -r requirements-pipeline.txt
```

### Configure

```bash
cp .env.example .env
# edit .env — set GCP_PROJECT to your own project
```

### Run the API locally

```bash
uvicorn api.dctii_api:app --reload --port 8080
# docs at http://localhost:8080/docs
```

### Train the model

```bash
python -m pipeline.run_predict_train
```

### Run tests

```bash
pytest -q
```

---

## API endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | public | Liveness + BigQuery connectivity probe |
| GET | `/api/v1/meta` | public | API version & metadata |
| GET | `/api/v1/map-sites` | public | Single payload powering the dashboard map |
| POST | `/api/v1/predict` | key | Predict thermal impact of a planned data center |
| GET | `/api/v1/sites` | key | Site registry |
| GET | `/api/v1/scores` | key | DCTII scores (filter by year/region) |
| GET | `/api/v1/indicators/{site_id}` | key | Per-indicator breakdown |
| GET | `/api/v1/causal/{site_id}` | key | Difference-in-differences causal results |
| GET | `/api/v1/delta-t/{site_id}` | key | Monthly day/night ΔT series |
| GET | `/api/v1/regions` | key | Regional roll-ups |
| GET | `/api/v1/summary` | key | Portfolio summary |

Protected endpoints require an `X-API-Key` header. With `DCTII_API_KEYS` unset, the service runs in **unauthenticated dev mode**. For production, source keys from Secret Manager (see `terraform/main.tf`). All BigQuery queries are parameterized; rate limiting defaults to 100 req/min/key.

---

## Model

- **Targets:** nighttime ΔT (primary production model) and daytime ΔT.
- **Algorithm:** stratified LightGBM quantile regressors (median + q10 + q90), routed by estimation method (CEM vs ring-difference).
- **Features:** 22 features in 5 tiers — physical (waste/sensible heat, PUE, capacity, footprint, load factor), biophysics (NDVI, vegetation deficit, impervious/tree/bare cover, elevation, population density), climate/cooling, interaction terms, and site context.
- **Validation:** Leave-One-Region-Out CV (5 folds); **Montreal held out entirely** as the most climatically distinct test region.
- **Uncertainty:** cross-conformal calibration on LORO holdouts (no train-set leakage); climate-stratified bias correction.

### v1 night-model results (held-out MTL test set)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| MAE | **0.276 °C** | ≤ 0.40 | ✅ |
| R² | **0.510** | ≥ 0.40 | ✅ |
| Pearson r | **0.854** | ≥ 0.65 | ✅ |
| CI coverage (90%) | **96.2 %** | ≥ 85 | ✅ |
| Spearman ρ | 0.630 | ≥ 0.70 | ⚠ conditional (n=26; 0.70 inside 95% CI) |

Daytime UHI has a low signal-to-noise ratio and remains challenging; the **night model is the production model**. See [`docs/DCTII_Predict_Project.md`](docs/DCTII_Predict_Project.md) for full methodology, SHAP drivers, and roadmap.

---

## Configuration

All runtime configuration is via environment variables (see [`.env.example`](.env.example)). The API reads `GCP_PROJECT`, the `BQ_*` dataset names, `DCTII_API_KEYS`, `RATE_LIMIT_PER_MINUTE`, `CORS_ORIGINS`, and `API_VERSION` from the environment, falling back to sensible defaults.

> **Note:** a few pipeline modules (`pipeline/pipeline_run.py`, `pipeline/populate_serving.py`) carry the author's GCP project ID (`oil-tank-monitoring-123`) as a constant default. A GCP project ID is not a secret, but change these to your own project before running the pipeline end-to-end.

---

## Infrastructure

`terraform/main.tf` provisions the full GCP footprint — BigQuery datasets/tables, Cloud Run service, GCS model bucket with IAM, a `pipeline_runs` lineage table, and a monitoring alert policy. Terraform **state files are intentionally not committed**; run `terraform init` to download providers (pinned in `.terraform.lock.hcl`).

---

## Data & reproducibility notes

- **No proprietary or third-party data is vendored.** Satellite inputs are pulled live from Earth Engine; the Köppen-Geiger raster is fetched by `scripts/download_koppen.py`; corporate sustainability reports used for the companion water-impact research are not redistributed.
- Trained model artifacts live in GCS (`gs://dctii-model-<project>/predict/v{N}/`), not in the repo.

---

## License

[MIT](LICENSE) © 2026 Oguzhan Tekin
