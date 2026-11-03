"""
evaluate.py - Standalone Model Evaluation & Classification Report Generator
Part of LungScan AI Project.
"""

import os
import sys
import json
import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.dirname(backend_dir))

from backend.ml.load_data import get_dataset_root, scan_dataset
from backend.ml.train import extract_dataset_features


def evaluate_saved_model(dataset_dir: str = None) -> dict:
    """
    Loads saved Logistic Regression model and computes unbiased metrics across verified scans.
    """
    model_path = os.path.join(backend_dir, "models", "lungscan_lr_pipeline.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained pipeline not found at {model_path}.")

    pipeline = joblib.load(model_path)
    root = get_dataset_root(dataset_dir)
    scan_res = scan_dataset(root)
    records = scan_res["records"]

    X_df, y, _ = extract_dataset_features(records)
    X = X_df.values

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, pos_label=1))
    rec = float(recall_score(y, y_pred, pos_label=1))
    f1 = float(f1_score(y, y_pred, pos_label=1))
    auc = float(roc_auc_score(y, y_proba))
    brier = float(brier_score_loss(y, y_proba))
    cm = confusion_matrix(y, y_pred).tolist()

    report_str = classification_report(y, y_pred, target_names=["Benign (0)", "Malignant (1)"])

    print("\n================ INDEPENDENT DATASET EVALUATION ================")
    print(report_str)
    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"ROC-AUC:         {auc:.4f}")
    print(f"Brier Loss:      {brier:.4f}")
    print(f"Confusion Matrix: {cm}")
    print("================================================================")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "brier_score": brier,
        "confusion_matrix": cm,
        "classification_report": report_str
    }


if __name__ == "__main__":
    evaluate_saved_model()

