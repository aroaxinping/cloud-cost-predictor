"""Tests for the FastAPI prediction endpoint."""
import numpy as np
import pytest

from src.api import app


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_models():
    """Mock the model loading so tests don't need model files."""
    class FakeModel:
        def predict(self, X):
            return np.full(len(X), 3.0)

    fake_models = {0.10: FakeModel(), 0.50: FakeModel(), 0.95: FakeModel()}
    fake_config = {
        "thresholds": {"terminate_cpu": 5, "downsize_cpu": 20, "review_cpu": 50},
        "risk_margins": {
            "terminate": {"safe": 3, "moderate": 1},
            "downsize": {"safe": 8, "moderate": 3},
        },
    }

    import src.api
    src.api._models = fake_models
    src.api._quantiles = [0.10, 0.50, 0.95]
    src.api._config = fake_config
    yield
    src.api._models = None
    src.api._quantiles = None
    src.api._config = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_single_vm(client):
    payload = {
        "vms": [{"instance": "vm-1", "cpu_std": 2.0, "cpu_min": 1.0,
                  "cpu_median": 3.0, "cv": 0.5}]
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) == 1
    pred = data["predictions"][0]
    assert pred["instance"] == "vm-1"
    assert pred["action"] == "terminate"


def test_predict_batch(client):
    vms = [
        {"instance": f"vm-{i}", "cpu_std": 2.0, "cpu_min": 1.0,
         "cpu_median": 3.0, "cv": 0.5}
        for i in range(5)
    ]
    r = client.post("/predict", json={"vms": vms})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 5


def test_predict_empty_batch(client):
    r = client.post("/predict", json={"vms": []})
    assert r.status_code == 422
