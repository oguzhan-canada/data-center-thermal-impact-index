terraform {
  required_version = ">= 1.6.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "project_id" {
  type    = string
  default = "oil-tank-monitoring-123"
}

variable "region" {
  type    = string
  default = "northamerica-northeast1"
}

variable "env" {
  type    = string
  default = "dev"
}

variable "dashboard_origin" {
  type        = string
  default     = "*"
  description = "Allowed CORS origin for the dashboard. Use '*' for dev, set to actual dashboard URL for prod."
}

variable "case_study_regions" {
  type    = list(string)
  default = ["phoenix", "houston", "nova", "central-tx", "toronto", "montreal"]
  description = "DCTII case study regions"
}

# ---------------------------------------------------------------------------
# Locals
# ---------------------------------------------------------------------------

locals {
  common_labels = {
    app         = "dctii"
    env         = var.env
    system      = "thermal-index"
    cost_center = "climate-analytics"
  }

  required_apis = toset([
    "artifactregistry.googleapis.com",
    "batch.googleapis.com",
    "bigquery.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudscheduler.googleapis.com",
    "earthengine.googleapis.com",
    "iam.googleapis.com",
    "logging.googleapis.com",
    "run.googleapis.com",
    "serviceusage.googleapis.com",
    "storage.googleapis.com",
  ])

  artifact_image_api      = "${var.region}-docker.pkg.dev/${var.project_id}/dctii-repo/dctii-api:${var.env}"
  artifact_image_pipeline = "${var.region}-docker.pkg.dev/${var.project_id}/dctii-repo/dctii-pipeline:${var.env}"
}

# ---------------------------------------------------------------------------
# Required APIs (reuses GOII pattern — most already enabled)
# ---------------------------------------------------------------------------

resource "google_project_service" "required" {
  for_each           = local.required_apis
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------------------
# Artifact Registry
# ---------------------------------------------------------------------------

resource "google_artifact_registry_repository" "dctii_repo" {
  location      = var.region
  repository_id = "dctii-repo"
  format        = "DOCKER"
  description   = "DCTII container images"
  labels        = local.common_labels
  depends_on    = [google_project_service.required]
}

# ---------------------------------------------------------------------------
# Cloud Storage buckets (4-tier, mirrors GOII)
# ---------------------------------------------------------------------------

resource "google_storage_bucket" "raw" {
  name                        = "dctii-raw-${var.project_id}"
  location                    = "NORTHAMERICA-NORTHEAST1"
  uniform_bucket_level_access = true
  force_destroy               = var.env == "dev" ? true : false
  labels                      = merge(local.common_labels, { data_tier = "raw" })

  lifecycle_rule {
    condition { age = 365 }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_storage_bucket" "processed" {
  name                        = "dctii-processed-${var.project_id}"
  location                    = "NORTHAMERICA-NORTHEAST1"
  uniform_bucket_level_access = true
  force_destroy               = var.env == "dev" ? true : false
  labels                      = merge(local.common_labels, { data_tier = "processed" })
  depends_on                  = [google_project_service.required]
}

resource "google_storage_bucket" "curated" {
  name                        = "dctii-curated-${var.project_id}"
  location                    = "NORTHAMERICA-NORTHEAST1"
  uniform_bucket_level_access = true
  force_destroy               = var.env == "dev" ? true : false
  labels                      = merge(local.common_labels, { data_tier = "curated" })
  depends_on                  = [google_project_service.required]
}

resource "google_storage_bucket" "model" {
  name                        = "dctii-model-${var.project_id}"
  location                    = "NORTHAMERICA-NORTHEAST1"
  uniform_bucket_level_access = true
  force_destroy               = var.env == "dev" ? true : false
  labels                      = merge(local.common_labels, { data_tier = "model" })
  depends_on                  = [google_project_service.required]
}

# ---------------------------------------------------------------------------
# BigQuery datasets (4-tier, mirrors GOII)
# ---------------------------------------------------------------------------

resource "google_bigquery_dataset" "ref" {
  dataset_id = "dctii_ref"
  location   = var.region
  labels     = merge(local.common_labels, { data_tier = "ref" })
  depends_on = [google_project_service.required]
}

resource "google_bigquery_dataset" "staging" {
  dataset_id = "dctii_staging"
  location   = var.region
  labels     = merge(local.common_labels, { data_tier = "staging" })
  depends_on = [google_project_service.required]
}

resource "google_bigquery_dataset" "curated" {
  dataset_id = "dctii_curated"
  location   = var.region
  labels     = merge(local.common_labels, { data_tier = "curated" })
  depends_on = [google_project_service.required]
}

resource "google_bigquery_dataset" "serving" {
  dataset_id = "dctii_serving"
  location   = var.region
  labels     = merge(local.common_labels, { data_tier = "serving" })
  depends_on = [google_project_service.required]
}

# ---------------------------------------------------------------------------
# BigQuery tables — ref layer (site registry + sensor metadata)
# ---------------------------------------------------------------------------

resource "google_bigquery_table" "site_registry" {
  dataset_id          = google_bigquery_dataset.ref.dataset_id
  table_id            = "site_registry"
  deletion_protection = var.env == "prod" ? true : false

  schema = jsonencode([
    { name = "site_id",          type = "STRING",  mode = "REQUIRED" },
    { name = "site_name",        type = "STRING",  mode = "NULLABLE" },
    { name = "operator",         type = "STRING",  mode = "NULLABLE" },
    { name = "latitude",         type = "FLOAT64", mode = "REQUIRED" },
    { name = "longitude",        type = "FLOAT64", mode = "REQUIRED" },
    { name = "country",          type = "STRING",  mode = "REQUIRED" },
    { name = "region_code",      type = "STRING",  mode = "REQUIRED" },
    { name = "climate_zone",     type = "STRING",  mode = "NULLABLE" },
    { name = "capacity_mw",      type = "FLOAT64", mode = "NULLABLE" },
    { name = "activation_year",  type = "INT64",   mode = "NULLABLE" },
    { name = "footprint_km2",    type = "FLOAT64", mode = "NULLABLE" },
    { name = "confidence_tier",  type = "INT64",   mode = "NULLABLE" },
    { name = "cluster_id",       type = "STRING",  mode = "NULLABLE" },
    { name = "pue_estimate",     type = "FLOAT64", mode = "NULLABLE" },
    { name = "pue_source",       type = "STRING",  mode = "NULLABLE",
      description = "How PUE was obtained: reported (operator/utility), ml_imputed (regression model), era_default (vintage-based default)" },
    { name = "load_factor",      type = "FLOAT64", mode = "NULLABLE",
      description = "Actual IT load as fraction of nameplate capacity_mw (default 0.6, range 0.4-0.8)" },
    { name = "cooling_type",     type = "STRING",  mode = "NULLABLE",
      description = "Cooling system: air_cooled, tower_cooled, unknown. Determines sensible/latent heat split." },
    { name = "created_ts",       type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  labels     = merge(local.common_labels, { data_tier = "ref" })
  depends_on = [google_bigquery_dataset.ref]
}

resource "google_bigquery_table" "sensor_metadata" {
  dataset_id          = google_bigquery_dataset.ref.dataset_id
  table_id            = "sensor_metadata"
  deletion_protection = var.env == "prod" ? true : false

  schema = jsonencode([
    { name = "sensor_id",        type = "STRING",  mode = "REQUIRED" },
    { name = "sensor_name",      type = "STRING",  mode = "REQUIRED" },
    { name = "spatial_res_m",    type = "FLOAT64", mode = "NULLABLE" },
    { name = "temporal_res",     type = "STRING",  mode = "NULLABLE" },
    { name = "overpass_time",    type = "STRING",  mode = "NULLABLE" },
    { name = "ee_collection",    type = "STRING",  mode = "NULLABLE" },
    { name = "band_name",        type = "STRING",  mode = "NULLABLE" },
    { name = "scale_factor",     type = "FLOAT64", mode = "NULLABLE" },
    { name = "active_from",      type = "DATE",    mode = "NULLABLE" },
    { name = "active_to",        type = "DATE",    mode = "NULLABLE" },
    { name = "reliable_obs_threshold",  type = "INT64", mode = "NULLABLE",
      description = "Min clear observations per month to classify as RELIABLE. Sensor-specific: MODIS=15, Landsat=2, ECOSTRESS=3" },
    { name = "low_count_obs_threshold", type = "INT64", mode = "NULLABLE",
      description = "Min clear observations per month to classify as LOW_COUNT (below reliable, above unreliable)" },
  ])

  labels     = merge(local.common_labels, { data_tier = "ref" })
  depends_on = [google_bigquery_dataset.ref]
}

# ---------------------------------------------------------------------------
# BigQuery tables — staging layer (raw LST extractions)
# ---------------------------------------------------------------------------

resource "google_bigquery_table" "lst_observations" {
  dataset_id          = google_bigquery_dataset.staging.dataset_id
  table_id            = "lst_observations"
  deletion_protection = var.env == "prod" ? true : false
  clustering          = ["region_code", "sensor_id"]

  time_partitioning {
    type  = "MONTH"
    field = "obs_date"
  }

  # Schema was recreated via BQ CREATE OR REPLACE TABLE during dedup.
  # Lifecycle ignore prevents Terraform from destroying the table with data.
  lifecycle {
    ignore_changes = [schema]
  }

  schema = jsonencode([
    { name = "obs_date",         type = "DATE",      mode = "REQUIRED" },
    { name = "site_id",          type = "STRING",    mode = "REQUIRED" },
    { name = "region_code",      type = "STRING",    mode = "REQUIRED" },
    { name = "sensor_id",        type = "STRING",    mode = "REQUIRED" },
    { name = "time_of_day",      type = "STRING",    mode = "REQUIRED" },
    { name = "zone_name",        type = "STRING",    mode = "NULLABLE",
      description = "Spatial extraction zone: footprint, near, buffer_1, buffer_2, control_near, control_far" },
    { name = "mean_lst_k",       type = "FLOAT64",   mode = "NULLABLE" },
    { name = "std_lst_k",        type = "FLOAT64",   mode = "NULLABLE" },
    { name = "pixel_count",      type = "INT64",     mode = "NULLABLE" },
    { name = "n_clear_obs",      type = "INT64",     mode = "NULLABLE",
      description = "Number of clear-sky observations in the monthly composite" },
    { name = "cloud_fraction",       type = "FLOAT64",   mode = "NULLABLE" },
    { name = "reliability_flag",     type = "STRING",    mode = "NULLABLE",
      description = "Sensor-specific quality: RELIABLE, LOW_COUNT, UNRELIABLE, or GAP_FILLED" },
    { name = "fill_method",          type = "STRING",    mode = "NULLABLE",
      description = "How this observation was produced: 'observed', 'ml_fusion' (ESTARFM/U-Net gap-fill), or 'interpolated'" },
    { name = "gap_fill_confidence",  type = "FLOAT64",   mode = "NULLABLE",
      description = "Confidence score [0-1] for gap-filled observations; NULL for directly observed" },
    { name = "gap_fill_model_version", type = "STRING",  mode = "NULLABLE",
      description = "Version of the gap-filling model used (e.g., 'unet_v1_phoenix'). Tracks which model weights produced this observation." },
    { name = "is_anomalous",         type = "BOOL",      mode = "NULLABLE",
      description = "True if flagged by LSTM autoencoder sensor-artifact detector" },
    { name = "anomaly_score",        type = "FLOAT64",   mode = "NULLABLE",
      description = "Reconstruction error from LSTM autoencoder; higher = more anomalous" },
    { name = "qa_flags",             type = "JSON",      mode = "NULLABLE" },
    { name = "created_ts",           type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  labels     = merge(local.common_labels, { data_tier = "staging" })
  depends_on = [google_bigquery_dataset.staging]
}

resource "google_bigquery_table" "control_observations" {
  dataset_id          = google_bigquery_dataset.staging.dataset_id
  table_id            = "control_observations"
  deletion_protection = var.env == "prod" ? true : false
  clustering          = ["region_code", "site_id"]

  time_partitioning {
    type  = "MONTH"
    field = "obs_date"
  }

  schema = jsonencode([
    { name = "obs_date",         type = "DATE",      mode = "REQUIRED" },
    { name = "site_id",          type = "STRING",    mode = "REQUIRED" },
    { name = "control_id",       type = "STRING",    mode = "REQUIRED" },
    { name = "region_code",      type = "STRING",    mode = "REQUIRED" },
    { name = "sensor_id",        type = "STRING",    mode = "REQUIRED" },
    { name = "time_of_day",      type = "STRING",    mode = "REQUIRED" },
    { name = "mean_lst_k",       type = "FLOAT64",   mode = "NULLABLE" },
    { name = "match_weight",     type = "FLOAT64",   mode = "NULLABLE" },
    { name = "created_ts",       type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  labels     = merge(local.common_labels, { data_tier = "staging" })
  depends_on = [google_bigquery_dataset.staging]
}

resource "google_bigquery_table" "site_candidates" {
  dataset_id          = google_bigquery_dataset.staging.dataset_id
  table_id            = "site_candidates"
  deletion_protection = var.env == "prod" ? true : false

  schema = jsonencode([
    { name = "candidate_id",        type = "STRING",    mode = "REQUIRED" },
    { name = "region_code",         type = "STRING",    mode = "REQUIRED" },
    { name = "latitude",            type = "FLOAT64",   mode = "REQUIRED" },
    { name = "longitude",           type = "FLOAT64",   mode = "REQUIRED" },
    { name = "detection_confidence", type = "FLOAT64",  mode = "NULLABLE",
      description = "CNN classifier confidence score [0-1]" },
    { name = "nighttime_anomaly_c", type = "FLOAT64",   mode = "NULLABLE",
      description = "Nighttime LST anomaly vs background (pre-filter >= 1.0 C)" },
    { name = "impervious_fraction", type = "FLOAT64",   mode = "NULLABLE",
      description = "Impervious surface fraction (pre-filter >= 0.6)" },
    { name = "detection_ts",        type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "review_status",       type = "STRING",    mode = "REQUIRED",
      description = "Workflow status: pending, approved, rejected" },
    { name = "reviewer_notes",      type = "STRING",    mode = "NULLABLE" },
    { name = "promoted_to_registry", type = "BOOL",     mode = "NULLABLE",
      description = "True after approved candidate is inserted into site_registry" },
  ])

  labels     = merge(local.common_labels, { data_tier = "staging" })
  depends_on = [google_bigquery_dataset.staging]
}

resource "google_bigquery_table" "site_covariates" {
  dataset_id          = google_bigquery_dataset.staging.dataset_id
  table_id            = "site_covariates"
  deletion_protection = var.env == "prod" ? true : false
  clustering          = ["region_code", "site_id"]

  time_partitioning {
    type  = "YEAR"
    field = "covariate_date"
  }

  schema = jsonencode([
    { name = "site_id",               type = "STRING",    mode = "REQUIRED" },
    { name = "region_code",           type = "STRING",    mode = "REQUIRED" },
    { name = "zone_name",             type = "STRING",    mode = "REQUIRED",
      description = "Ring zone: footprint, near, buffer_1, buffer_2, control_near, control_far" },
    { name = "covariate_date",        type = "DATE",      mode = "REQUIRED",
      description = "Reference date for temporal covariates (Jan 1 of year)" },
    { name = "ndvi_max",              type = "FLOAT64",   mode = "NULLABLE",
      description = "Annual max NDVI from Landsat growing-season composite" },
    { name = "ndvi_mean",             type = "FLOAT64",   mode = "NULLABLE",
      description = "Annual mean NDVI" },
    { name = "impervious_fraction",   type = "FLOAT64",   mode = "NULLABLE",
      description = "Fraction of zone with impervious surface (ESA WorldCover built-up class)" },
    { name = "tree_cover_fraction",   type = "FLOAT64",   mode = "NULLABLE",
      description = "Fraction of zone with tree cover" },
    { name = "bare_fraction",         type = "FLOAT64",   mode = "NULLABLE",
      description = "Fraction of zone with bare/sparse vegetation" },
    { name = "water_fraction",        type = "FLOAT64",   mode = "NULLABLE",
      description = "Fraction of zone with permanent water" },
    { name = "cropland_fraction",     type = "FLOAT64",   mode = "NULLABLE",
      description = "Fraction of zone with cropland" },
    { name = "population_density",    type = "FLOAT64",   mode = "NULLABLE",
      description = "Mean population density (persons/km2) from WorldPop" },
    { name = "population_total",      type = "FLOAT64",   mode = "NULLABLE",
      description = "Total population within zone from WorldPop" },
    { name = "elevation_mean_m",      type = "FLOAT64",   mode = "NULLABLE",
      description = "Mean elevation (meters) from SRTM 30m" },
    { name = "elevation_std_m",       type = "FLOAT64",   mode = "NULLABLE",
      description = "Elevation standard deviation (meters)" },
    { name = "snow_cover_days",       type = "FLOAT64",   mode = "NULLABLE",
      description = "Annual snow-covered days from MODIS MOD10A2 (Canada only)" },
    { name = "valid_pixel_fraction",  type = "FLOAT64",   mode = "NULLABLE",
      description = "Fraction of zone with valid data (QA check)" },
    { name = "source_products",       type = "STRING",    mode = "NULLABLE",
      description = "Comma-separated list of source datasets used" },
    { name = "created_ts",            type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  labels     = merge(local.common_labels, { data_tier = "staging" })
  depends_on = [google_bigquery_dataset.staging]
}

resource "google_bigquery_table" "ndvi_timeseries" {
  dataset_id          = google_bigquery_dataset.staging.dataset_id
  table_id            = "ndvi_timeseries"
  deletion_protection = var.env == "prod" ? true : false

  time_partitioning {
    type  = "YEAR"
    field = "obs_date"
  }

  clustering = ["region_code", "site_id"]

  schema = jsonencode([
    { name = "site_id",       type = "STRING",    mode = "REQUIRED",
      description = "FK to site_registry" },
    { name = "region_code",   type = "STRING",    mode = "REQUIRED",
      description = "Region code (PHX, HOU, NOVA, CTX, TOR, MTL)" },
    { name = "zone_name",     type = "STRING",    mode = "REQUIRED",
      description = "Ring zone (footprint, near, buffer_1, buffer_2, control_near, control_far)" },
    { name = "year",          type = "INT64",     mode = "REQUIRED",
      description = "Calendar year of growing-season NDVI composite" },
    { name = "obs_date",      type = "DATE",      mode = "REQUIRED",
      description = "Partition key (YYYY-01-01 for annual)" },
    { name = "ndvi_max",      type = "FLOAT64",   mode = "NULLABLE",
      description = "Growing-season maximum NDVI (Landsat composite Apr-Sep)" },
    { name = "ndvi_mean",     type = "FLOAT64",   mode = "NULLABLE",
      description = "Growing-season mean NDVI" },
    { name = "ndvi_p25",      type = "FLOAT64",   mode = "NULLABLE",
      description = "25th percentile NDVI in zone" },
    { name = "ndvi_p75",      type = "FLOAT64",   mode = "NULLABLE",
      description = "75th percentile NDVI in zone" },
    { name = "valid_pixel_count", type = "INT64", mode = "NULLABLE",
      description = "Number of valid Landsat pixels in zone" },
    { name = "source_collection", type = "STRING", mode = "NULLABLE",
      description = "GEE collection ID (Landsat 8/9)" },
    { name = "created_ts",    type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  labels     = merge(local.common_labels, { data_tier = "staging" })
  depends_on = [google_bigquery_dataset.staging]
}

resource "google_bigquery_table" "cem_matches" {
  dataset_id          = google_bigquery_dataset.staging.dataset_id
  table_id            = "cem_matches"
  deletion_protection = var.env == "prod" ? true : false

  clustering = ["estimation_unit_id", "treated_site_id"]

  schema = jsonencode([
    { name = "match_id",              type = "STRING",    mode = "REQUIRED",
      description = "Unique match record ID" },
    { name = "estimation_unit_id",    type = "STRING",    mode = "REQUIRED",
      description = "Estimation unit (site_id or cluster ID for SUTVA)" },
    { name = "treated_site_id",       type = "STRING",    mode = "REQUIRED",
      description = "FK to site_registry" },
    { name = "treated_zone_name",     type = "STRING",    mode = "REQUIRED",
      description = "Zone of treated unit" },
    { name = "control_site_id",       type = "STRING",    mode = "REQUIRED",
      description = "Control site FK" },
    { name = "control_zone_name",     type = "STRING",    mode = "REQUIRED",
      description = "Zone of control unit" },
    { name = "cohort_year",           type = "INT64",     mode = "REQUIRED",
      description = "Treatment cohort year (activation_year or cluster rule)" },
    { name = "matching_strategy",     type = "STRING",    mode = "REQUIRED",
      description = "CEM, propensity, or synthetic_control" },
    { name = "cem_stratum",           type = "STRING",    mode = "NULLABLE",
      description = "Coarsened stratum key (e.g., ndvi_q2_imp_q3_elev_300)" },
    { name = "cem_weight",            type = "FLOAT64",   mode = "REQUIRED",
      description = "CEM weight for this match pair" },
    { name = "ndvi_treated",          type = "FLOAT64",   mode = "NULLABLE",
      description = "Pre-treatment NDVI of treated zone" },
    { name = "ndvi_control",          type = "FLOAT64",   mode = "NULLABLE",
      description = "Pre-treatment NDVI of control zone" },
    { name = "impervious_treated",    type = "FLOAT64",   mode = "NULLABLE",
      description = "Impervious fraction of treated zone" },
    { name = "impervious_control",    type = "FLOAT64",   mode = "NULLABLE",
      description = "Impervious fraction of control zone" },
    { name = "elevation_treated",     type = "FLOAT64",   mode = "NULLABLE",
      description = "Elevation (m) of treated zone" },
    { name = "elevation_control",     type = "FLOAT64",   mode = "NULLABLE",
      description = "Elevation (m) of control zone" },
    { name = "pre_lst_mean_treated",  type = "FLOAT64",   mode = "NULLABLE",
      description = "Pre-treatment mean LST of treated zone" },
    { name = "pre_lst_mean_control",  type = "FLOAT64",   mode = "NULLABLE",
      description = "Pre-treatment mean LST of control zone" },
    { name = "pre_lst_trend_treated", type = "FLOAT64",   mode = "NULLABLE",
      description = "Pre-treatment LST trend (C/year) of treated zone" },
    { name = "pre_lst_trend_control", type = "FLOAT64",   mode = "NULLABLE",
      description = "Pre-treatment LST trend (C/year) of control zone" },
    { name = "smd_ndvi",              type = "FLOAT64",   mode = "NULLABLE",
      description = "Standardized mean difference for NDVI" },
    { name = "smd_impervious",        type = "FLOAT64",   mode = "NULLABLE",
      description = "Standardized mean difference for impervious fraction" },
    { name = "smd_elevation",         type = "FLOAT64",   mode = "NULLABLE",
      description = "Standardized mean difference for elevation" },
    { name = "smd_pre_lst",           type = "FLOAT64",   mode = "NULLABLE",
      description = "Standardized mean difference for pre-treatment LST" },
    { name = "match_version",         type = "INT64",     mode = "REQUIRED",
      description = "Match run version (monotonic)" },
    { name = "created_ts",            type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  labels     = merge(local.common_labels, { data_tier = "staging" })
  depends_on = [google_bigquery_dataset.staging]
}

# ---------------------------------------------------------------------------
# BigQuery tables — curated layer (anomalies + indicators)
# ---------------------------------------------------------------------------

resource "google_bigquery_table" "delta_t_monthly" {
  dataset_id          = google_bigquery_dataset.curated.dataset_id
  table_id            = "delta_t_monthly"
  deletion_protection = var.env == "prod" ? true : false

  time_partitioning {
    type  = "MONTH"
    field = "year_month"
  }

  schema = jsonencode([
    { name = "year_month",       type = "DATE",      mode = "REQUIRED" },
    { name = "site_id",          type = "STRING",    mode = "REQUIRED" },
    { name = "sensor_id",        type = "STRING",    mode = "REQUIRED" },
    { name = "delta_t_day_c",    type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_night_c",  type = "FLOAT64",   mode = "NULLABLE" },
    { name = "n_clear_days",     type = "INT64",     mode = "NULLABLE" },
    { name = "n_clear_nights",   type = "INT64",     mode = "NULLABLE" },
    { name = "n_gap_filled",     type = "INT64",     mode = "NULLABLE",
      description = "Number of gap-filled observations used in this monthly composite" },
    { name = "reliability_flag", type = "STRING",    mode = "NULLABLE",
      description = "Sensor-specific quality flag: RELIABLE, LOW_COUNT, UNRELIABLE, or GAP_FILLED. Thresholds vary by sensor revisit rate - see sensor_metadata table." },
    { name = "r_effective_km",   type = "FLOAT64",   mode = "NULLABLE" },
    { name = "season",           type = "STRING",    mode = "NULLABLE" },
    { name = "created_ts",       type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  labels     = merge(local.common_labels, { data_tier = "curated" })
  depends_on = [google_bigquery_dataset.curated]
}

resource "google_bigquery_table" "site_indicators" {
  dataset_id          = google_bigquery_dataset.curated.dataset_id
  table_id            = "site_indicators"
  deletion_protection = var.env == "prod" ? true : false

  schema = jsonencode([
    { name = "year",                    type = "INT64",     mode = "REQUIRED" },
    { name = "site_id",                 type = "STRING",    mode = "REQUIRED" },
    { name = "delta_t_day",             type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_night",           type = "FLOAT64",   mode = "NULLABLE" },
    { name = "heat_island_area_km2",    type = "FLOAT64",   mode = "NULLABLE" },
    { name = "population_exposed",      type = "FLOAT64",   mode = "NULLABLE" },
    { name = "waste_heat_flux_wm2",     type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_day_norm",        type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_night_norm",      type = "FLOAT64",   mode = "NULLABLE" },
    { name = "heat_island_area_norm",   type = "FLOAT64",   mode = "NULLABLE" },
    { name = "population_exposed_norm", type = "FLOAT64",   mode = "NULLABLE" },
    { name = "waste_heat_flux_norm",    type = "FLOAT64",   mode = "NULLABLE" },
    { name = "min_monthly_reliability", type = "STRING",   mode = "NULLABLE",
      description = "Worst reliability_flag among the 12 months contributing to this annual indicator (RELIABLE/LOW_COUNT/UNRELIABLE/GAP_FILLED)" },
    { name = "fraction_reliable_months", type = "FLOAT64", mode = "NULLABLE",
      description = "Fraction of months with reliability_flag=RELIABLE (0.0-1.0). Indicates overall data quality for this site-year." },
    { name = "created_ts",              type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  labels     = merge(local.common_labels, { data_tier = "curated" })
  depends_on = [google_bigquery_dataset.curated]
}

# ---------------------------------------------------------------------------
# BigQuery tables — serving layer (DCTII scores + DiD results)
# ---------------------------------------------------------------------------

resource "google_bigquery_table" "dctii_scores" {
  dataset_id          = google_bigquery_dataset.serving.dataset_id
  table_id            = "dctii_scores"
  deletion_protection = var.env == "prod" ? true : false

  time_partitioning {
    type  = "MONTH"
    field = "created_ts"
  }

  schema = jsonencode([
    { name = "year",              type = "INT64",     mode = "REQUIRED" },
    { name = "site_id",           type = "STRING",    mode = "REQUIRED" },
    { name = "weighting_scheme",  type = "STRING",    mode = "REQUIRED" },
    { name = "dctii_score",       type = "FLOAT64",   mode = "NULLABLE" },
    { name = "impact_category",   type = "STRING",    mode = "NULLABLE" },
    { name = "ci_lower",          type = "FLOAT64",   mode = "NULLABLE" },
    { name = "ci_upper",          type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_day",       type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_night",     type = "FLOAT64",   mode = "NULLABLE" },
    { name = "heat_island_area_km2", type = "FLOAT64", mode = "NULLABLE" },
    { name = "population_exposed",   type = "FLOAT64", mode = "NULLABLE" },
    { name = "waste_heat_flux_wm2",  type = "FLOAT64", mode = "NULLABLE" },
    { name = "data_version",      type = "INT64",     mode = "REQUIRED",
      description = "Monotonic version counter; increments when normalization bounds or site registry changes" },
    { name = "norm_bounds_hash",  type = "STRING",    mode = "NULLABLE",
      description = "SHA-256 of normalization min/max bounds used for this scoring run" },
    { name = "created_ts",        type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  labels     = merge(local.common_labels, { data_tier = "serving" })
  depends_on = [google_bigquery_dataset.serving]
}

resource "google_bigquery_table" "did_results" {
  dataset_id          = google_bigquery_dataset.serving.dataset_id
  table_id            = "did_results"
  deletion_protection = var.env == "prod" ? true : false

  schema = jsonencode([
    { name = "site_id",           type = "STRING",    mode = "REQUIRED" },
    { name = "cohort_year",       type = "INT64",     mode = "REQUIRED" },
    { name = "outcome_var",       type = "STRING",    mode = "REQUIRED" },
    { name = "att_estimate",      type = "FLOAT64",   mode = "NULLABLE" },
    { name = "att_se",            type = "FLOAT64",   mode = "NULLABLE" },
    { name = "att_ci_lower",      type = "FLOAT64",   mode = "NULLABLE" },
    { name = "att_ci_upper",      type = "FLOAT64",   mode = "NULLABLE" },
    { name = "p_value",           type = "FLOAT64",   mode = "NULLABLE" },
    { name = "pre_trend_f_stat",  type = "FLOAT64",   mode = "NULLABLE" },
    { name = "pre_trend_p_value", type = "FLOAT64",   mode = "NULLABLE" },
    { name = "n_treated",         type = "INT64",     mode = "NULLABLE" },
    { name = "n_control",         type = "INT64",     mode = "NULLABLE" },
    { name = "se_method",         type = "STRING",    mode = "NULLABLE" },
    { name = "estimation_unit",   type = "STRING",    mode = "NULLABLE",
      description = "Unit of observation: 'site' for isolated DCs, 'cluster' for cluster-as-unit (NoVA)" },
    { name = "data_version",      type = "INT64",     mode = "REQUIRED",
      description = "Matches data_version in dctii_scores for provenance tracking" },
    { name = "created_ts",        type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  labels     = merge(local.common_labels, { data_tier = "serving" })
  depends_on = [google_bigquery_dataset.serving]
}

# ---------------------------------------------------------------------------
# Service accounts (mirrors GOII pattern + least-privilege API SA)
# ---------------------------------------------------------------------------

resource "google_service_account" "runner" {
  account_id   = "sa-dctii-runner-${var.env}"
  display_name = "DCTII Pipeline Runner (${var.env})"
}

resource "google_service_account" "scheduler" {
  account_id   = "sa-dctii-scheduler-${var.env}"
  display_name = "DCTII Scheduler Invoker (${var.env})"
}

resource "google_service_account" "api_reader" {
  account_id   = "sa-dctii-api-${var.env}"
  display_name = "DCTII API Read-Only Service Account (${var.env})"
}

# Runner SA permissions (pipeline: read/write BQ + GCS + EE)
resource "google_project_iam_member" "runner_bq" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.runner.email}"
}

resource "google_project_iam_member" "runner_gcs" {
  project = var.project_id
  role    = "roles/storage.objectCreator"
  member  = "serviceAccount:${google_service_account.runner.email}"
}

resource "google_project_iam_member" "runner_gcs_reader" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.runner.email}"
}

resource "google_project_iam_member" "runner_ee" {
  project = var.project_id
  role    = "roles/earthengine.writer"
  member  = "serviceAccount:${google_service_account.runner.email}"
}

# Runner SA permissions for Cloud Batch (ML training)
resource "google_project_iam_member" "runner_batch" {
  project = var.project_id
  role    = "roles/batch.jobsEditor"
  member  = "serviceAccount:${google_service_account.runner.email}"
}

resource "google_service_account_iam_member" "runner_sa_self_impersonate" {
  service_account_id = google_service_account.runner.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.runner.email}"
}

# API SA permissions (read-only: only needs dataViewer on serving dataset)
resource "google_bigquery_dataset_iam_member" "api_reader_serving" {
  dataset_id = google_bigquery_dataset.serving.dataset_id
  role       = "roles/bigquery.dataViewer"
  member     = "serviceAccount:${google_service_account.api_reader.email}"
}

resource "google_project_iam_member" "api_reader_bq_job" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.api_reader.email}"
}

# ---------------------------------------------------------------------------
# Cloud Run — Pipeline Job (batch LST processing)
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_job" "pipeline" {
  name                = "job-dctii-pipeline-${var.env}"
  location            = var.region
  labels              = local.common_labels
  deletion_protection = var.env == "prod" ? true : false

  template {
    template {
      service_account = google_service_account.runner.email
      timeout         = "3600s"
      max_retries     = 3

      containers {
        image = local.artifact_image_pipeline

        resources {
          limits = {
            cpu    = "4"
            memory = "8Gi"
          }
        }

        env {
          name  = "GCP_PROJECT"
          value = var.project_id
        }
        env {
          name  = "DCTII_ENV"
          value = var.env
        }
      }
    }
  }

  depends_on = [google_project_service.required]
}

# ---------------------------------------------------------------------------
# Cloud Run — API Service (FastAPI)
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "api" {
  name                = "svc-dctii-api-${var.env}"
  location            = var.region
  labels              = local.common_labels
  deletion_protection = var.env == "prod" ? true : false

  template {
    service_account = google_service_account.api_reader.email

    containers {
      image = local.artifact_image_api

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }
      env {
        name  = "BQ_SERVING_DATASET"
        value = "dctii_serving"
      }
      env {
        name  = "CORS_ORIGINS"
        value = var.dashboard_origin
      }
      # DCTII_API_KEYS: In dev, omitted = unauthenticated dev mode (intentional).
      # For staging/prod, source from Secret Manager:
      #   env { name = "DCTII_API_KEYS"; value_source { secret_key_ref { ... } } }
    }
  }

  depends_on = [google_project_service.required]
}

# Allow unauthenticated access — application-level API key auth handles security.
# Without this, Cloud Run's proxy returns 403 before FastAPI even sees the request.
resource "google_cloud_run_v2_service_iam_member" "api_public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.api.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ---------------------------------------------------------------------------
# Cloud Scheduler — Monthly pipeline trigger
# ---------------------------------------------------------------------------

resource "google_cloud_scheduler_job" "monthly_pipeline" {
  name      = "dctii-monthly-pipeline-${var.env}"
  region    = var.region
  schedule  = "0 6 1 * *"
  time_zone = "America/Toronto"

  http_target {
    http_method = "POST"
    uri         = "https://${var.region}-run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.pipeline.name}:run"

    oauth_token {
      service_account_email = google_service_account.scheduler.email
    }
  }

  depends_on = [google_cloud_run_v2_job.pipeline]
}

# Scheduler SA needs invoker permission to trigger pipeline job via HTTP
resource "google_cloud_run_v2_job_iam_member" "scheduler_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_job.pipeline.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.scheduler.email}"
}

# ---------------------------------------------------------------------------
# Cloud Batch — ML Training Job (GPU workers for gap-filling, anomaly detection)
# ---------------------------------------------------------------------------
# ML training (U-Net gap-filler, LSTM anomaly detector, CNN site detector, PUE
# regressor) requires GPU and runs longer than Cloud Run's 60-min timeout.
# Cloud Batch is the first-class infrastructure for this, NOT optional.
# Alternative: Vertex AI Training Jobs (if more managed experience is preferred).

# ═══════════════════════════════════════════════════════════════════════════════
# DCTII-Predict — ML Module 6: New Resources
# ═══════════════════════════════════════════════════════════════════════════════

# --- BQ Table: prediction_features (audit trail for GEE-extracted features) ---
resource "google_bigquery_table" "prediction_features" {
  dataset_id          = google_bigquery_dataset.staging.dataset_id
  table_id            = "prediction_features"
  deletion_protection = false

  schema = jsonencode([
    { name = "prediction_id",      type = "STRING",    mode = "REQUIRED" },
    { name = "created_ts",         type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "latitude",           type = "FLOAT64",   mode = "REQUIRED" },
    { name = "longitude",          type = "FLOAT64",   mode = "REQUIRED" },
    { name = "climate_zone",       type = "STRING",    mode = "NULLABLE" },
    { name = "ndvi_growing_max",   type = "FLOAT64",   mode = "NULLABLE" },
    { name = "impervious_fraction", type = "FLOAT64",  mode = "NULLABLE" },
    { name = "tree_cover_fraction", type = "FLOAT64",  mode = "NULLABLE" },
    { name = "bare_fraction",      type = "FLOAT64",   mode = "NULLABLE" },
    { name = "population_density", type = "FLOAT64",   mode = "NULLABLE" },
    { name = "elevation_m",        type = "FLOAT64",   mode = "NULLABLE" },
    { name = "snow_cover_days",    type = "FLOAT64",   mode = "NULLABLE" },
    { name = "gee_status",         type = "STRING",    mode = "NULLABLE" },
    { name = "extraction_year",    type = "INT64",     mode = "NULLABLE" },
  ])
}

# --- BQ Table: predictions (full prediction outputs — queryable audit) ---
resource "google_bigquery_table" "predictions" {
  dataset_id          = google_bigquery_dataset.serving.dataset_id
  table_id            = "predictions"
  deletion_protection = false

  time_partitioning {
    type  = "MONTH"
    field = "created_ts"
  }

  schema = jsonencode([
    { name = "prediction_id",            type = "STRING",    mode = "REQUIRED" },
    { name = "created_ts",               type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "latitude",                 type = "FLOAT64",   mode = "REQUIRED" },
    { name = "longitude",                type = "FLOAT64",   mode = "REQUIRED" },
    { name = "climate_zone",             type = "STRING",    mode = "NULLABLE" },
    { name = "capacity_mw",              type = "FLOAT64",   mode = "REQUIRED" },
    { name = "cooling_type",             type = "STRING",    mode = "REQUIRED" },
    { name = "pue_used",                 type = "FLOAT64",   mode = "NULLABLE" },
    { name = "pue_was_imputed",          type = "BOOL",      mode = "NULLABLE" },
    { name = "footprint_km2",            type = "FLOAT64",   mode = "NULLABLE" },
    { name = "load_factor",              type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_day_pred",         type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_night_pred",       type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_day_ci_lower",     type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_day_ci_upper",     type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_night_ci_lower",   type = "FLOAT64",   mode = "NULLABLE" },
    { name = "delta_t_night_ci_upper",   type = "FLOAT64",   mode = "NULLABLE" },
    { name = "heat_island_area_km2",     type = "FLOAT64",   mode = "NULLABLE" },
    { name = "population_exposed",       type = "FLOAT64",   mode = "NULLABLE" },
    { name = "waste_heat_flux_wm2",      type = "FLOAT64",   mode = "NULLABLE" },
    { name = "dctii_score",              type = "FLOAT64",   mode = "NULLABLE" },
    { name = "impact_category",          type = "STRING",    mode = "NULLABLE" },
    { name = "distribution_shift_score", type = "FLOAT64",   mode = "NULLABLE" },
    { name = "distribution_shift_label", type = "STRING",    mode = "NULLABLE" },
    { name = "model_version",            type = "STRING",    mode = "NULLABLE" },
    { name = "weighting_scheme",         type = "STRING",    mode = "NULLABLE" },
    { name = "gee_status",               type = "STRING",    mode = "NULLABLE" },
  ])
}

# --- GCS IAM: API SA needs GCS read for model artifacts ---
resource "google_storage_bucket_iam_member" "api_model_reader" {
  bucket = google_storage_bucket.model.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api_reader.email}"
}

# --- Cloud Monitoring: predict endpoint p99 latency alert ---
resource "google_monitoring_alert_policy" "predict_latency" {
  display_name = "DCTII Predict API p99 latency alert"
  combiner     = "OR"

  conditions {
    display_name = "predict_p99_latency"
    condition_threshold {
      filter          = "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_latencies\""
      comparison      = "COMPARISON_GT"
      threshold_value = 8000
      duration        = "120s"

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_PERCENTILE_99"
      }
    }
  }

  notification_channels = []
}
# Note: batch.googleapis.com is included in local.required_apis.

# ML model storage bucket (separate from raster storage)
resource "google_storage_bucket" "ml_models" {
  name          = "dctii-models-${var.project_id}"
  location      = var.region
  storage_class = "STANDARD"
  labels        = merge(local.common_labels, { purpose = "ml-models" })

  versioning {
    enabled = true  # Track model weight versions for gap_fill_model_version provenance
  }

  uniform_bucket_level_access = true

  depends_on = [google_project_service.required]
}

# Runner SA needs objectAdmin on ml_models bucket for Batch job read/write
resource "google_storage_bucket_iam_member" "runner_ml_models_admin" {
  bucket = google_storage_bucket.ml_models.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runner.email}"
}

# Validation station reference table (for train/val/test split)
resource "google_bigquery_table" "validation_stations" {
  dataset_id          = google_bigquery_dataset.ref.dataset_id
  table_id            = "validation_stations"
  deletion_protection = var.env == "prod" ? true : false

  schema = jsonencode([
    { name = "station_id",     type = "STRING",  mode = "REQUIRED" },
    { name = "station_name",   type = "STRING",  mode = "NULLABLE" },
    { name = "source",         type = "STRING",  mode = "REQUIRED",
      description = "Data source: NOAA_ASOS, NRCAN_CLIMATE, EPA_AQS, ECOSTRESS" },
    { name = "latitude",       type = "FLOAT64", mode = "REQUIRED" },
    { name = "longitude",      type = "FLOAT64", mode = "REQUIRED" },
    { name = "region_code",    type = "STRING",  mode = "REQUIRED" },
    { name = "split_group",    type = "STRING",  mode = "REQUIRED",
      description = "Three-way split: 'train' (60%, weight optimization), 'validation' (20%, tuning), 'test' (20%, final RMSE report only)" },
    { name = "elevation_m",    type = "FLOAT64", mode = "NULLABLE" },
    { name = "active_from",    type = "DATE",    mode = "NULLABLE" },
    { name = "active_to",      type = "DATE",    mode = "NULLABLE" },
    { name = "winter_reliable", type = "BOOL",   mode = "NULLABLE",
      description = "False for Canadian stations with unreliable winter LST over snow cover" },
  ])

  labels     = merge(local.common_labels, { data_tier = "ref" })
  depends_on = [google_bigquery_dataset.ref]
}

# ---------------------------------------------------------------------------
# Pipeline Run Tracking Table (C-04, R-01)
# ---------------------------------------------------------------------------

resource "google_bigquery_table" "pipeline_runs" {
  dataset_id          = google_bigquery_dataset.staging.dataset_id
  table_id            = "pipeline_runs"
  deletion_protection = false

  time_partitioning {
    type  = "DAY"
    field = "started_at"
  }

  clustering = ["stage", "year"]

  schema = jsonencode([
    { name = "run_id",        type = "STRING",    mode = "REQUIRED" },
    { name = "stage",         type = "STRING",    mode = "REQUIRED" },
    { name = "year",          type = "INT64",     mode = "NULLABLE" },
    { name = "status",        type = "STRING",    mode = "REQUIRED" },
    { name = "started_at",    type = "TIMESTAMP", mode = "REQUIRED" },
    { name = "completed_at",  type = "TIMESTAMP", mode = "NULLABLE" },
    { name = "rows_written",  type = "INT64",     mode = "NULLABLE" },
    { name = "error_message", type = "STRING",    mode = "NULLABLE" },
    { name = "triggered_by",  type = "STRING",    mode = "NULLABLE" },
    { name = "force_rerun",   type = "BOOL",      mode = "NULLABLE" },
  ])

  labels     = merge(local.common_labels, { data_tier = "staging" })
  depends_on = [google_bigquery_dataset.staging]
}

# ---------------------------------------------------------------------------
# Pipeline Failure Monitoring (R-02)
# ---------------------------------------------------------------------------

variable "alert_email" {
  type        = string
  description = "Email address for pipeline failure alerts (leave empty to disable)"
  default     = ""
}

resource "google_monitoring_notification_channel" "pipeline_alerts" {
  count        = var.alert_email != "" ? 1 : 0
  display_name = "DCTII Pipeline Alerts"
  type         = "email"
  labels       = { email_address = var.alert_email }
  project      = var.project_id
}

resource "google_monitoring_alert_policy" "pipeline_failure" {
  count        = var.alert_email != "" ? 1 : 0
  display_name = "DCTII pipeline job failure"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "Cloud Run Job task failed"
    condition_threshold {
      filter = join(" AND ", [
        "metric.type=\"run.googleapis.com/job/completed_task_count\"",
        "resource.labels.job_name=\"job-dctii-pipeline-${var.env}\"",
        "metric.labels.result=\"failed\"",
      ])
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }

  notification_channels = google_monitoring_notification_channel.pipeline_alerts[*].id

  alert_strategy {
    auto_close = "604800s"
  }
}
