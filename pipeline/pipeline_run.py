"""
Pipeline run tracking for idempotency and lineage (C-04, H-08, L-06).

Creates and manages a `pipeline_runs` control table in BigQuery to:
  - Prevent duplicate runs (idempotency guard)
  - Track run status, timing, and row counts
  - Enable lineage via pipeline_run_id in downstream tables

DDL must be run once in BigQuery before first use:

    CREATE TABLE IF NOT EXISTS `{PROJECT}.dctii_staging.pipeline_runs` (
        run_id STRING NOT NULL,
        stage STRING NOT NULL,           -- 'anomaly_compute', 'populate_serving'
        year INT64,
        status STRING NOT NULL,          -- 'running', 'success', 'failed', 'skipped'
        started_at TIMESTAMP NOT NULL,
        completed_at TIMESTAMP,
        rows_written INT64,
        error_message STRING,
        triggered_by STRING,             -- 'manual', 'scheduler', 'backfill'
        force_rerun BOOL DEFAULT FALSE,
    );

Usage:
    from pipeline.pipeline_run import PipelineRunTracker

    tracker = PipelineRunTracker(client, stage="anomaly_compute", year=2024)
    if not tracker.start(force=args.force):
        print("Already completed — skipping (use --force to override)")
        return
    try:
        n = do_work()
        tracker.complete(rows_written=n)
    except Exception as e:
        tracker.fail(str(e))
        raise
"""

import uuid
import logging
from datetime import datetime, timezone
from google.cloud import bigquery

logger = logging.getLogger("dctii.pipeline_run")

PROJECT = "oil-tank-monitoring-123"
RUNS_TABLE = f"{PROJECT}.dctii_staging.pipeline_runs"


class PipelineRunTracker:
    """Track pipeline run status for idempotency and lineage."""

    def __init__(self, client: bigquery.Client, stage: str, year: int = None,
                 triggered_by: str = "manual"):
        self.client = client
        self.stage = stage
        self.year = year
        self.triggered_by = triggered_by
        self.run_id = uuid.uuid4().hex
        self._started = False

    def _table_exists(self) -> bool:
        """Check if pipeline_runs table exists."""
        try:
            self.client.get_table(RUNS_TABLE)
            return True
        except Exception:
            return False

    def has_successful_run(self) -> bool:
        """Check if a successful run already exists for this stage+year."""
        if not self._table_exists():
            return False
        sql = f"""
        SELECT COUNT(*) as cnt FROM `{RUNS_TABLE}`
        WHERE stage = @stage AND status = 'success'
        """
        params = [bigquery.ScalarQueryParameter("stage", "STRING", self.stage)]
        if self.year:
            sql += " AND year = @year"
            params.append(bigquery.ScalarQueryParameter("year", "INT64", self.year))
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        rows = list(self.client.query(sql, job_config=cfg).result())
        return rows[0].cnt > 0

    def start(self, force: bool = False) -> bool:
        """Register run start. Returns False if should skip (idempotent guard)."""
        if not force and self.has_successful_run():
            logger.info(f"Skipping {self.stage} year={self.year} — already successful")
            return False

        if not self._table_exists():
            # R-08: Clear warning when table is absent (bootstrap scenario)
            logger.warning(
                f"pipeline_runs table not found — run tracking disabled. "
                f"Apply Terraform to create dctii_staging.pipeline_runs before production use."
            )
            self._started = True
            return True

        # Table exists — insert run record
        sql = f"""
        INSERT INTO `{RUNS_TABLE}`
        (run_id, stage, year, status, started_at, triggered_by, force_rerun)
        VALUES (@run_id, @stage, @year, 'running', CURRENT_TIMESTAMP(), @triggered_by, @force)
        """
        params = [
            bigquery.ScalarQueryParameter("run_id", "STRING", self.run_id),
            bigquery.ScalarQueryParameter("stage", "STRING", self.stage),
            bigquery.ScalarQueryParameter("year", "INT64", self.year),
            bigquery.ScalarQueryParameter("triggered_by", "STRING", self.triggered_by),
            bigquery.ScalarQueryParameter("force", "BOOL", force),
        ]
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(sql, job_config=cfg).result()

        self._started = True
        logger.info(f"Pipeline run started: {self.run_id} ({self.stage}, year={self.year})")
        return True

    def complete(self, rows_written: int = 0):
        """Mark run as successful."""
        if not self._started or not self._table_exists():
            return
        sql = f"""
        UPDATE `{RUNS_TABLE}`
        SET status = 'success', completed_at = CURRENT_TIMESTAMP(), rows_written = @rows
        WHERE run_id = @run_id
        """
        params = [
            bigquery.ScalarQueryParameter("run_id", "STRING", self.run_id),
            bigquery.ScalarQueryParameter("rows", "INT64", rows_written),
        ]
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(sql, job_config=cfg).result()
        logger.info(f"Pipeline run complete: {self.run_id} ({rows_written} rows)")

    def fail(self, error_message: str):
        """Mark run as failed."""
        if not self._started or not self._table_exists():
            return
        sql = f"""
        UPDATE `{RUNS_TABLE}`
        SET status = 'failed', completed_at = CURRENT_TIMESTAMP(),
            error_message = @error
        WHERE run_id = @run_id
        """
        params = [
            bigquery.ScalarQueryParameter("run_id", "STRING", self.run_id),
            bigquery.ScalarQueryParameter("error", "STRING", error_message[:2000]),
        ]
        cfg = bigquery.QueryJobConfig(query_parameters=params)
        self.client.query(sql, job_config=cfg).result()
        logger.error(f"Pipeline run failed: {self.run_id} — {error_message[:200]}")
