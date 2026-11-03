"""
test_api.py - Unit and Integration Tests for LungScan AI Backend
"""

import os
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["pipeline_loaded"] is True
    assert data["metadata_loaded"] is True


def test_model_metrics():
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "evaluation_metrics" in data
    assert "accuracy" in data["evaluation_metrics"]
    assert "roc_auc" in data["evaluation_metrics"]
    assert "confusion_matrix" in data
    assert "roc_curve" in data
    assert "pr_curve" in data
    assert data["evaluation_metrics"]["accuracy"] > 0.90


def test_model_info():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "LungScan AI Logistic Regression Classifier"
    assert "Logistic Regression" in data["algorithm"]
    assert "feature_importance" in data
    assert len(data["feature_names"]) == 21


def test_classes():
    response = client.get("/api/classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data
    assert data["classes"]["0"]["name"] == "Benign"
    assert data["classes"]["1"]["name"] == "Malignant"


def test_get_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2


def test_predict_with_sample_benign():
    response = client.post("/api/predict", data={"sample_id": "benign_1"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["predicted_class"] == "Benign"
    assert data["predicted_label"] == 0
    assert "probabilities" in data
    assert data["probabilities"]["benign"] > 0.50
    assert len(data["extracted_features"]) == 21


def test_predict_with_sample_malignant():
    response = client.post("/api/predict", data={"sample_id": "malignant_1"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["predicted_class"] == "Malignant"
    assert data["predicted_label"] == 1
    assert "probabilities" in data
    assert data["probabilities"]["malignant"] > 0.50
    assert len(data["extracted_features"]) == 21


def test_invalid_file_extension():
    files = {"file": ("test.pdf", b"%PDF-1.4 dummy", "application/pdf")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

