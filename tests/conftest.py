"""Shared test fixtures for DCTII tests (L-02)."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd


@pytest.fixture
def mock_bq_client():
    """Mock BigQuery client that returns empty results by default."""
    with patch("google.cloud.bigquery.Client") as mock:
        client = MagicMock()
        mock.return_value = client
        # Default: queries return empty DataFrame
        client.query.return_value.to_dataframe.return_value = pd.DataFrame()
        client.query.return_value.result.return_value = iter([])
        yield client


@pytest.fixture
def mock_empty_bq(mock_bq_client):
    """BQ client that returns empty results for all queries."""
    mock_bq_client.query.return_value.to_dataframe.return_value = pd.DataFrame()
    return mock_bq_client
