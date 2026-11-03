"""
model_info.py - FastAPI routes for model metadata, evaluation metrics, and sample scans.
Part of LungScan AI Project.
"""

import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api", tags=["Model Information"])

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
models_dir = os.path.join(backend_dir, "models")
samples_dir = os.path.join(backend_dir, "samples")


def load_metadata_cache():
    meta_path = os.path.join(models_dir, "model_metadata.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=500, detail="Model metadata not found. Please train model first.")
    with open(meta_path, "r") as f:
        return json.load(f)


@router.get("/health")
def health_check():
    """Health status and system verification."""
    pipeline_exists = os.path.exists(os.path.join(models_dir, "lungscan_lr_pipeline.joblib"))
    meta_exists = os.path.exists(os.path.join(models_dir, "model_metadata.json"))
    return {
        "status": "healthy" if (pipeline_exists and meta_exists) else "degraded",
        "service": "LungScan AI Backend",
        "pipeline_loaded": pipeline_exists,
        "metadata_loaded": meta_exists,
        "framework": "FastAPI + Scikit-Learn",
    }


@router.get("/model/metrics")
def get_model_metrics():
    """Returns actual held-out test set evaluation metrics and curve data."""
    meta = load_metadata_cache()
    return {
        "evaluation_metrics": meta["evaluation_metrics"],
        "confusion_matrix": meta["confusion_matrix"],
        "roc_curve": meta["roc_curve"],
        "pr_curve": meta["pr_curve"],
        "hyperparameters": meta["hyperparameters"],
    }


@router.get("/model/info")
def get_model_info():
    """Returns dataset origin, feature list, and model coefficients."""
    meta = load_metadata_cache()
    return {
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
        "algorithm": meta["algorithm"],
        "training_date": meta["training_date"],
        "dataset": meta["dataset"],
        "feature_names": meta["feature_names"],
        "feature_importance": meta["feature_importance"],
        "clinical_disclaimer": meta["clinical_disclaimer"],
    }


@router.get("/classes")
def get_classes():
    """Returns the primary binary classification targets."""
    meta = load_metadata_cache()
    return {
        "binary_task": "Benign vs Malignant Lung CT Scan Classification",
        "classes": meta.get("classes", {
            "0": {"name": "Benign", "label": 0},
            "1": {"name": "Malignant", "label": 1}
        })
    }


@router.get("/samples")
def get_sample_scans():
    """Returns list of curated test CT scans for immediate frontend testing."""
    manifest_path = os.path.join(samples_dir, "samples.json")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r") as f:
        return json.load(f)


@router.get("/samples/image/{filename}")
def get_sample_image(filename: str):
    """Serves the actual image bytes for curated sample scans."""
    # Prevent directory traversal
    clean_filename = os.path.basename(filename)
    img_path = os.path.join(samples_dir, clean_filename)
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    return FileResponse(img_path, media_type="image/jpeg")

