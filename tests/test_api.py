"""
Tests for the FastAPI endpoints.
Uses TestClient to call the API in-process — no Docker needed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_model():
    """
    Create a TestClient with a mocked model loaded.
    Patches BOTH the model finder AND mlflow loader to bypass real file lookup.
    """
    with patch("src.api.main.find_champion_model") as mock_find, \
         patch("src.api.main.mlflow.pyfunc.load_model") as mock_load:
        # Make the finder return a fake path (doesn't need to exist — mock_load handles it)
        mock_find.return_value = "/fake/path/to/model"
        
        # Mock model that returns a probability of 0.42
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.42]
        mock_load.return_value = mock_model

        from src.api.main import app
        with TestClient(app) as client:
            yield client


@pytest.fixture
def client_without_model():
    """
    Create a TestClient where model loading fails.
    """
    with patch("src.api.main.find_champion_model") as mock_find:
        mock_find.return_value = None  # Simulates model not found
        
        from src.api.main import app
        with TestClient(app) as client:
            yield client


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client_with_model):
        response = client_with_model.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_when_model_loaded(self, client_with_model):
        response = client_with_model.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_includes_timestamp(self, client_with_model):
        response = client_with_model.get("/health")
        data = response.json()
        assert "timestamp" in data


class TestRootEndpoint:
    """Tests for the / endpoint."""

    def test_root_returns_200(self, client_with_model):
        response = client_with_model.get("/")
        assert response.status_code == 200

    def test_root_returns_service_info(self, client_with_model):
        response = client_with_model.get("/")
        data = response.json()
        assert "service" in data
        assert "status" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_with_valid_payload(self, client_with_model):
        payload = {
            "customer_id": "TEST_001",
            "features": {
                "P_2_mean": 0.45,
                "B_1_max": 0.82,
                "S_3_last": 0.31,
            }
        }
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == "TEST_001"
        assert 0.0 <= data["default_probability"] <= 1.0
        assert data["risk_tier"] in ["low", "medium", "high"]

    def test_predict_rejects_missing_customer_id(self, client_with_model):
        payload = {"features": {"P_2_mean": 0.5}}
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 422  # Pydantic validation error

    def test_predict_rejects_missing_features(self, client_with_model):
        payload = {"customer_id": "TEST_002"}
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_returns_high_tier_for_high_probability(self, client_with_model):
        # Mock returns 0.42 → medium tier
        payload = {
            "customer_id": "TEST_003",
            "features": {"feature_1": 0.5}
        }
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_tier"] == "medium"  # 0.42 falls in medium range