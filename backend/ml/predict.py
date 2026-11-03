"""
predict.py - Real-Time Inference and Radiomic Feature Explainability
Part of LungScan AI Project.
"""

import os
import sys
import json
from typing import Dict, Any, Union, List
import numpy as np
import joblib

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.dirname(backend_dir))

from backend.ml.preprocess import preprocess_pipeline
from backend.ml.feature_extraction import (
    extract_features,
    extract_feature_vector,
    get_feature_metadata,
    ORDERED_FEATURE_NAMES
)

# Global lazy-loaded artifacts
_PIPELINE = None
_METADATA = None


def get_model_artifacts():
    """Lazy loads and caches the trained pipeline and metadata."""
    global _PIPELINE, _METADATA
    if _PIPELINE is None:
        model_path = os.path.join(backend_dir, "models", "lungscan_lr_pipeline.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model pipeline not found at {model_path}. Please train the model first.")
        _PIPELINE = joblib.load(model_path)

    if _METADATA is None:
        meta_path = os.path.join(backend_dir, "models", "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                _METADATA = json.load(f)
        else:
            _METADATA = {}

    return _PIPELINE, _METADATA


def predict_ct_scan(
    image_input: Union[str, bytes, np.ndarray],
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Executes end-to-end prediction on a CT scan image:
    1. Preprocessing (grayscale, standard resize, bilateral/Gaussian denoise, CLAHE, lung masking)
    2. Radiomic feature extraction (21 statistical, GLCM, LBP, and edge metrics)
    3. StandardScaler transformation + Logistic Regression inference
    4. Feature contribution breakdown (logit decomposition)
    5. Formulates clinical explanation and educational disclaimer
    """
    pipeline, metadata = get_model_artifacts()
    
    # 1. Preprocess
    pre = preprocess_pipeline(image_input)
    
    # 2. Extract features
    features_dict = extract_features(pre)
    X_vec = np.array([[features_dict[k] for k in ORDERED_FEATURE_NAMES]], dtype=np.float32)

    # 3. Model Inference
    # The pipeline consists of [('scaler', StandardScaler()), ('classifier', LogisticRegression())]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]

    X_scaled = scaler.transform(X_vec)[0] # 1D scaled vector
    coefs = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])

    # Raw log-odds (logit)
    logit = float(np.dot(X_scaled, coefs) + intercept)
    prob_malignant = float(1.0 / (1.0 + np.exp(-logit)))
    prob_benign = float(1.0 - prob_malignant)

    predicted_label = 1 if prob_malignant >= threshold else 0
    predicted_class = "Malignant" if predicted_label == 1 else "Benign"
    confidence = prob_malignant if predicted_label == 1 else prob_benign

    # 4. Feature Contributions to Log-Odds
    feature_meta = get_feature_metadata()
    contributions = []
    for name, raw_val, scaled_val, coef in zip(ORDERED_FEATURE_NAMES, X_vec[0], X_scaled, coefs):
        impact = float(scaled_val * coef)
        contributions.append({
            "feature": name,
            "raw_value": round(float(raw_val), 4),
            "z_score": round(float(scaled_val), 3),
            "coefficient": round(float(coef), 3),
            "logit_impact": round(impact, 3),
            "direction": "Towards Malignant" if impact > 0 else "Towards Benign",
            "description": feature_meta.get(name, "")
        })

    # Sort contributions by absolute logit impact
    contributions.sort(key=lambda x: abs(x["logit_impact"]), reverse=True)

    # 5. Explanatory Narrative
    top_push = [c for c in contributions if (c["logit_impact"] > 0 if predicted_label == 1 else c["logit_impact"] < 0)][:3]
    top_feature_names = ", ".join([f"{c['feature']} ({c['description'].split(' ')[0]})" for c in top_push])
    
    if predicted_label == 1:
        narrative = (
            f"The Logistic Regression model estimated an {confidence * 100:.1f}% probability of a malignant neoplasm. "
            f"Key driving indicators include elevated {top_feature_names}, reflecting increased tissue heterogeneity, "
            f"spiculated edge characteristics, and abnormal CT attenuation within thoracic fields."
        )
    else:
        narrative = (
            f"The Logistic Regression model estimated an {confidence * 100:.1f}% probability of a benign lesion. "
            f"Key driving indicators include {top_feature_names}, reflecting regular micro-texture uniformity, "
            f"lower edge spiculation, and lower attenuation gradients."
        )

    return {
        "status": "success",
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "benign": round(prob_benign, 4),
            "malignant": round(prob_malignant, 4)
        },
        "threshold_applied": threshold,
        "raw_logit": round(logit, 4),
        "model_version": metadata.get("model_version", "1.0.0"),
        "training_date": metadata.get("training_date", ""),
        "extracted_features": features_dict,
        "feature_contributions": contributions,
        "top_contributions": contributions[:6],
        "explanation": narrative,
        "preprocessed_stats": {
            "raw_shape": list(pre["raw_shape"]),
            "target_size": list(pre["target_size"]),
            "parenchyma_ratio": round(pre["parenchyma_ratio"], 4)
        },
        "disclaimer": (
            "This result is produced by an experimental machine-learning research model. "
            "It is NOT a medical diagnosis and must not be used to make treatment or clinical decisions."
        )
    }


if __name__ == "__main__":
    # Test on one of the sample images
    sample_file = os.path.join(backend_dir, "samples", "sample_malignant_1.jpg")
    res = predict_ct_scan(sample_file)
    print("Test inference prediction:")
    print("Class:", res["predicted_class"])
    print("Probabilities:", res["probabilities"])
    print("Explanation:", res["explanation"])

