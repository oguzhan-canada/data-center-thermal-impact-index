"""
DCTII-Predict Integration Tests — mocked GEE and BQ.

Tests the /api/v1/predict endpoint end-to-end with mock model artifacts.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from pipeline.predict_train import FEATURE_COLUMNS


# ── Fixtures ──────────────────────────────────────────────────────────────

PHOENIX = {
    "latitude": 33.4484,
    "longitude": -112.0740,
    "capacity_mw": 50.0,
    "cooling_type": "air_cooled",
    "pue": 1.25,
    "load_factor": 0.6,
}

MOCK_GEE_FEATURES = {
    "ndvi_growing_max": 0.15,
    "impervious_fraction": 0.65,
    "tree_cover_fraction": 0.05,
    "bare_fraction": 0.30,
    "population_density": 320.0,
    "elevation_m": 340.0,
    "snow_cover_days": 0.0,
    "extraction_year": 2024,
    "gee_status": "ok",
}


class FakeModel:
    """Minimal mock that mimics LGBMRegressor.predict()."""
    def __init__(self, value=0.8):
        self._value = value

    def predict(self, X):
        return np.array([self._value] * X.shape[0])


def _make_mock_models():
    """Build mock model artifacts dict matching load_model_artifacts() shape."""
    n_features = len(FEATURE_COLUMNS)
    return {
        "day_cem_median": FakeModel(0.8),
        "day_cem_q10": FakeModel(0.4),
        "day_cem_q90": FakeModel(1.2),
        "day_ring_median": FakeModel(0.6),
        "day_ring_q10": FakeModel(0.3),
        "day_ring_q90": FakeModel(1.0),
        "night_median": FakeModel(0.5),
        "night_q10": FakeModel(0.2),
        "night_q90": FakeModel(0.9),
        "corrections": {
            "day_cem_correction": 0.1,
            "day_ring_correction": 0.1,
            "night_correction": 0.1,
            "night_bias_offsets": {5: 0.02, "global": 0.01},
            "day_cem_bias_offsets": {5: 0.03, "global": 0.02},
        },
        "distribution": {
            "centroid": [0.0] * n_features,
            "cov_inv": np.eye(n_features).tolist(),
            "n_training": 280,
        },
        "eval_report": {"version": "v1"},
        "shap_day_cem": MagicMock(
            **{"shap_values.return_value": np.zeros((1, n_features))}
        ),
        "shap_night": MagicMock(
            **{"shap_values.return_value": np.zeros((1, n_features))}
        ),
        "feature_meta": {"feature_columns": FEATURE_COLUMNS},
    }


@pytest.fixture(autouse=True)
def mock_models():
    """All integration tests use mocked models — no GCS download."""
    with patch("api.dctii_api.get_predict_models") as m:
        m.return_value = _make_mock_models()
        yield m


@pytest.fixture(autouse=True)
def mock_gee():
    """All integration tests use mocked GEE — no real Earth Engine calls."""
    with patch("api.dctii_api.extract_predict_features") as m:
        m.return_value = MOCK_GEE_FEATURES.copy()
        yield m


@pytest.fixture(autouse=True)
def mock_bq_write():
    """Prevent BQ writes during tests."""
    with patch("api.dctii_api._write_prediction_to_bq"):
        yield


@pytest.fixture(autouse=True)
def mock_bq_health():
    """Mock BQ health check."""
    with patch("api.dctii_api.get_bq_client") as m:
        mock_client = MagicMock()
        mock_client.query.return_value.result.return_value = None
        m.return_value = mock_client
        yield


@pytest.fixture
def client():
    from api.dctii_api import app
    return TestClient(app)


HEADERS = {"X-API-Key": "test-key", "Content-Type": "application/json"}


# ── Tests ─────────────────────────────────────────────────────────────────

def test_predict_phoenix_returns_200(client):
    resp = client.post("/api/v1/predict", json=PHOENIX, headers=HEADERS)
    assert resp.status_code == 200


def test_predict_response_schema(client):
    """All required fields present with correct types."""
    body = client.post("/api/v1/predict", json=PHOENIX, headers=HEADERS).json()
    assert isinstance(body["dctii_score"], float)
    assert isinstance(body["delta_t_night_c"], float)
    assert isinstance(body["delta_t_day_c"], float)
    assert isinstance(body["prediction_id"], str)
    assert len(body["prediction_id"]) == 36  # UUID format
    assert body["impact_category"] in [
        "Minimal", "Low", "Moderate", "High", "Severe"
    ]
    assert body["model_version"] == "v1"
    assert body["day_prediction_method"] in ["cem_primary", "ring_difference_discounted", "derived_from_night"]
    assert isinstance(body["cool_island_risk"], bool)


def test_predict_ci_lower_less_than_upper(client):
    body = client.post("/api/v1/predict", json=PHOENIX, headers=HEADERS).json()
    assert body["delta_t_night_ci"][0] <= body["delta_t_night_ci"][1]
    assert body["delta_t_day_ci"][0] <= body["delta_t_day_ci"][1]


def test_predict_physical_floors(client):
    """ΔT values must never be negative."""
    body = client.post("/api/v1/predict", json=PHOENIX, headers=HEADERS).json()
    assert body["delta_t_night_c"] >= 0.0
    assert body["delta_t_day_c"] >= 0.0
    assert body["delta_t_night_ci"][0] >= 0.0
    assert body["delta_t_day_ci"][0] >= 0.0


def test_predict_pue_imputation_flag(client):
    """Omitting PUE sets pue_was_imputed=True."""
    payload = {k: v for k, v in PHOENIX.items() if k != "pue"}
    body = client.post("/api/v1/predict", json=payload, headers=HEADERS).json()
    assert body["pue_was_imputed"] is True


def test_predict_explicit_pue_not_imputed(client):
    body = client.post("/api/v1/predict", json=PHOENIX, headers=HEADERS).json()
    assert body["pue_was_imputed"] is False


def test_predict_missing_location_returns_422(client):
    payload = {"capacity_mw": 50, "cooling_type": "air_cooled"}
    resp = client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert resp.status_code == 422


def test_predict_invalid_capacity_returns_422(client):
    resp = client.post(
        "/api/v1/predict",
        json={**PHOENIX, "capacity_mw": -10},
        headers=HEADERS,
    )
    assert resp.status_code == 422


def test_predict_shap_top5_structure(client):
    body = client.post("/api/v1/predict", json=PHOENIX, headers=HEADERS).json()
    for key in ["shap_night_top5", "shap_day_top5"]:
        assert len(body[key]) == 5
        for item in body[key]:
            assert "feature" in item
            assert "shap_impact" in item
            assert "value" in item


def test_predict_gee_timeout_uses_fallback(client, mock_gee):
    """GEE timeout falls back gracefully — no 500 error."""
    mock_gee.side_effect = TimeoutError("GEE timeout")
    resp = client.post("/api/v1/predict", json=PHOENIX, headers=HEADERS)
    assert resp.status_code == 200
    assert resp.json()["gee_extraction_status"] == "climate_fallback"


def test_predict_distribution_shift_in_response(client):
    """Response should include distribution shift assessment."""
    body = client.post("/api/v1/predict", json=PHOENIX, headers=HEADERS).json()
    assert "distribution_shift_score" in body
    assert "distribution_shift_label" in body
    assert body["distribution_shift_label"] in [
        "in_distribution", "moderate_shift", "high_shift", "extrapolation"
    ]


def test_predict_invalid_pue_returns_422(client):
    """PUE below 1.0 should return 422."""
    resp = client.post(
        "/api/v1/predict",
        json={**PHOENIX, "pue": 0.5},
        headers=HEADERS,
    )
    assert resp.status_code == 422


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "predict_models" in body
