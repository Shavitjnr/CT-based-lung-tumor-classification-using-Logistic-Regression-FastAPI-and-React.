"""
train.py - Stratified Training, Hyperparameter Tuning, and Evaluation of Logistic Regression
Part of LungScan AI Project.
"""

import os
import sys
import json
import time
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    classification_report
)

# Local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))
from backend.ml.load_data import get_dataset_root, scan_dataset
from backend.ml.preprocess import preprocess_pipeline
from backend.ml.feature_extraction import (
    extract_features,
    get_feature_names,
    get_feature_metadata,
    ORDERED_FEATURE_NAMES
)


def extract_dataset_features(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Extracts radiomic features from each verified image record in the dataset.
    Returns:
        df_features: DataFrame of numerical features
        y: target labels (0=Benign, 1=Malignant)
        filepaths: list of image paths
    """
    print(f"Extracting handcrafted features for {len(records)} CT scans...")
    features_list = []
    labels = []
    filepaths = []
    
    start_time = time.time()
    for idx, rec in enumerate(records):
        try:
            pre = preprocess_pipeline(rec["filepath"])
            feats = extract_features(pre)
            features_list.append(feats)
            labels.append(rec["class_label"])
            filepaths.append(rec["filepath"])
        except Exception as e:
            print(f"Warning: skipped {rec['filepath']}: {e}")

        if (idx + 1) % 100 == 0 or (idx + 1) == len(records):
            elapsed = time.time() - start_time
            print(f"  Processed {idx + 1}/{len(records)} images ({elapsed:.1f}s)")

    df_features = pd.DataFrame(features_list, columns=ORDERED_FEATURE_NAMES)
    y = np.array(labels, dtype=int)
    return df_features, y, filepaths


def run_training_pipeline(dataset_dir: str = None) -> Dict[str, Any]:
    """
    Executes the complete machine learning workflow:
    1. Dataset loading & deduplication
    2. Feature extraction
    3. Stratified Train-Test split (80/20)
    4. Stratified 5-Fold Cross-Validation on Training data with GridSearchCV
    5. Final evaluation on unseen Test set
    6. Extraction of ROC, PR curves, Confusion Matrix, and Feature Coefficients
    7. Artifact persistence (Joblib + JSON)
    """
    root = get_dataset_root(dataset_dir)
    scan_res = scan_dataset(root)
    records = scan_res["records"]

    if len(records) == 0:
        raise ValueError("No valid CT scan images found for training.")

    X_df, y, filepaths = extract_dataset_features(records)
    X = X_df.values
    feature_names = ORDERED_FEATURE_NAMES

    print(f"\nFeature Matrix Shape: {X.shape}, Class distribution: Benign={np.sum(y == 0)}, Malignant={np.sum(y == 1)}")

    # 1. Stratified Train/Test Split (Preserve test set for final unbiased evaluation)
    X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
        X, y, filepaths,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print(f"Train split: {len(y_train)} (Benign={np.sum(y_train == 0)}, Malignant={np.sum(y_train == 1)})")
    print(f"Test split:  {len(y_test)} (Benign={np.sum(y_test == 0)}, Malignant={np.sum(y_test == 1)})")

    # 2. Base Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 3. Hyperparameter Tuning using Stratified 5-Fold Cross Validation strictly on training set
    param_grid = [
        # L2 Regularization (lbfgs)
        {
            "classifier__penalty": ["l2"],
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "classifier__solver": ["lbfgs"],
        },
        # L1 Regularization (liblinear)
        {
            "classifier__penalty": ["l1"],
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            "classifier__solver": ["liblinear"],
        }
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=True
    )

    print("\nRunning Stratified 5-Fold Cross-Validation & Grid Search...")
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = float(grid_search.best_score_)
    print(f"Best CV ROC-AUC: {best_cv_score:.4f} with params: {best_params}")

    # Extract Cross-Validation Fold details
    cv_results = grid_search.cv_results_
    best_idx = grid_search.best_index_
    cv_mean_train_score = float(cv_results["mean_train_score"][best_idx])
    cv_std_test_score = float(cv_results["std_test_score"][best_idx])

    # 4. Final Evaluation on Held-Out Test Set
    y_test_pred = best_pipeline.predict(X_test)
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = float(accuracy_score(y_test, y_test_pred))
    prec = float(precision_score(y_test, y_test_pred, pos_label=1))
    rec = float(recall_score(y_test, y_test_pred, pos_label=1)) # Sensitivity for malignant
    f1 = float(f1_score(y_test, y_test_pred, pos_label=1))
    roc_auc = float(roc_auc_score(y_test, y_test_proba))
    brier = float(brier_score_loss(y_test, y_test_proba))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0 # Specificity for benign

    print(f"\n================ FINAL TEST SET EVALUATION ================")
    print(f"Accuracy:                {acc * 100:.2f}%")
    print(f"Malignant Precision:     {prec * 100:.2f}%")
    print(f"Malignant Sensitivity:   {rec * 100:.2f}% (Recall)")
    print(f"Benign Specificity:      {spec * 100:.2f}%")
    print(f"F1-Score:                {f1:.4f}")
    print(f"ROC-AUC:                 {roc_auc:.4f}")
    print(f"Brier Calibration Loss:  {brier:.4f}")
    print(f"Confusion Matrix:        TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("===========================================================")

    # 5. ROC Curve Data for visualization
    fpr, tpr, roc_thresh = roc_curve(y_test, y_test_proba)
    roc_points = []
    # Sample down to ~30 clean points for smooth frontend plotting
    step = max(1, len(fpr) // 30)
    for i in range(0, len(fpr), step):
        t_val = float(roc_thresh[i]) if i < len(roc_thresh) else 0.0
        if np.isinf(t_val) or np.isnan(t_val) or t_val > 1.0:
            t_val = 1.0
        roc_points.append({
            "fpr": round(float(fpr[i]), 4),
            "tpr": round(float(tpr[i]), 4),
            "threshold": round(t_val, 4)
        })
    if roc_points[-1]["fpr"] != 1.0 or roc_points[-1]["tpr"] != 1.0:
        roc_points.append({"fpr": 1.0, "tpr": 1.0, "threshold": 0.0})

    # 6. Precision-Recall Curve Data
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_test, y_test_proba)
    pr_points = []
    step_pr = max(1, len(pr_rec) // 30)
    for i in range(0, len(pr_rec), step_pr):
        t_val = float(pr_thresh[i]) if i < len(pr_thresh) else 0.0
        if np.isinf(t_val) or np.isnan(t_val) or t_val > 1.0:
            t_val = 1.0
        pr_points.append({
            "recall": round(float(pr_rec[i]), 4),
            "precision": round(float(pr_prec[i]), 4),
            "threshold": round(t_val, 4)
        })

    # 7. Model Coefficients / Feature Weights
    lr_model = best_pipeline.named_steps["classifier"]
    coeffs = lr_model.coef_[0]
    intercept = float(lr_model.intercept_[0])
    
    feature_importance = []
    feat_meta = get_feature_metadata()
    for name, coef in zip(feature_names, coeffs):
        feature_importance.append({
            "feature": name,
            "coefficient": round(float(coef), 4),
            "abs_weight": round(float(abs(coef)), 4),
            "odds_ratio": round(float(np.exp(coef)), 4),
            "direction": "Pushes Malignant" if coef > 0 else "Pushes Benign",
            "description": feat_meta.get(name, "")
        })
    feature_importance.sort(key=lambda x: x["abs_weight"], reverse=True)

    # 8. Save Artifacts
    models_dir = os.path.join(os.path.dirname(current_dir), "models")
    os.makedirs(models_dir, exist_ok=True)
    pipeline_save_path = os.path.join(models_dir, "lungscan_lr_pipeline.joblib")
    joblib.dump(best_pipeline, pipeline_save_path)
    print(f"\nTrained Pipeline saved to: {pipeline_save_path}")

    # Copy sample images for frontend demonstration
    samples_dir = os.path.join(os.path.dirname(current_dir), "samples")
    os.makedirs(samples_dir, exist_ok=True)
    sample_manifest = []

    # Select 3 benign and 3 malignant images from test set
    benign_samples = [p for p, label in zip(paths_test, y_test) if label == 0][:3]
    malignant_samples = [p for p, label in zip(paths_test, y_test) if label == 1][:3]

    for i, p in enumerate(benign_samples):
        dst_name = f"sample_benign_{i + 1}.jpg"
        dst_path = os.path.join(samples_dir, dst_name)
        shutil.copyfile(p, dst_path)
        sample_manifest.append({
            "id": f"benign_{i + 1}",
            "filename": dst_name,
            "true_class": "Benign",
            "true_label": 0,
            "description": f"Verified Benign Case #{i + 1} (Thoracic CT)",
            "source_file": os.path.basename(p)
        })

    for i, p in enumerate(malignant_samples):
        dst_name = f"sample_malignant_{i + 1}.jpg"
        dst_path = os.path.join(samples_dir, dst_name)
        shutil.copyfile(p, dst_path)
        sample_manifest.append({
            "id": f"malignant_{i + 1}",
            "filename": dst_name,
            "true_class": "Malignant",
            "true_label": 1,
            "description": f"Verified Malignant Case #{i + 1} (Thoracic CT)",
            "source_file": os.path.basename(p)
        })

    with open(os.path.join(samples_dir, "samples.json"), "w") as f:
        json.dump(sample_manifest, f, indent=2)

    # Metadata dictionary
    metadata = {
        "model_name": "LungScan AI Logistic Regression Classifier",
        "model_version": "1.0.0",
        "algorithm": "Logistic Regression with L1/L2 Regularization",
        "training_date": datetime.now().isoformat(),
        "dataset": {
            "name": "IQ-OTH/NCCD Lung Cancer Dataset",
            "source_institution": "Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases",
            "modality": "Chest CT (Siemens SOMATOM Scanner)",
            "total_binary_images": len(records),
            "benign_count": scan_res["benign_count"],
            "malignant_count": scan_res["malignant_count"],
            "normal_count_excluded": scan_res["normal_count"],
            "duplicates_removed": scan_res["duplicates_count"]
        },
        "hyperparameters": {
            "best_params": best_params,
            "class_weight": "balanced",
            "max_iter": 3000,
            "intercept": round(intercept, 4)
        },
        "evaluation_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall_sensitivity": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "brier_score": round(brier, 4),
            "cv_mean_roc_auc": round(best_cv_score, 4),
            "cv_std_roc_auc": round(cv_std_test_score, 4),
            "cv_mean_train_roc_auc": round(cv_mean_train_score, 4)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "total_test": len(y_test),
            "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
        },
        "roc_curve": roc_points,
        "pr_curve": pr_points,
        "feature_importance": feature_importance,
        "feature_names": feature_names,
        "classes": {
            "0": {
                "name": "Benign",
                "label": 0,
                "description": "Non-cancerous pulmonary nodule or benign inflammatory lesion",
                "clinical_guidance": "Usually presents with well-circumscribed borders, homogeneous attenuation, and lower spatial entropy."
            },
            "1": {
                "name": "Malignant",
                "label": 1,
                "description": "Cancerous lung neoplasm / pulmonary carcinoma",
                "clinical_guidance": "Characterized by spiculation, micro-texture irregularity, heterogeneous CT attenuation, and invasiveness."
            }
        },
        "clinical_disclaimer": "This experimental machine-learning prototype is developed for educational and academic research evaluation. It does NOT constitute medical advice, diagnosis, or treatment planning."
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Model Metadata & Real Evaluation Metrics saved to: {metadata_path}")

    return metadata


if __name__ == "__main__":
    run_training_pipeline()

