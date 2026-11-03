"""
main.py - LungScan AI Orchestrator & CLI Runner
Project: Lung Tumor Classification Using Logistic Regression and CT Scan Image Analysis
"""

import os
import sys
import argparse
import kagglehub

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def download_dataset():
    """Downloads the IQ-OTH/NCCD Lung Cancer Dataset using kagglehub."""
    print("\n--- Downloading IQ-OTH/NCCD Lung Cancer Dataset via kagglehub ---")
    path = kagglehub.dataset_download("hamdallak/the-iqothnccd-lung-cancer-dataset")
    print("Path to dataset files:", path)
    return path


def train_model():
    """Executes the machine learning training pipeline."""
    from backend.ml.train import run_training_pipeline
    print("\n--- Running Logistic Regression Training Pipeline ---")
    metadata = run_training_pipeline()
    return metadata


def evaluate_model():
    """Runs independent evaluation on the trained pipeline."""
    from backend.ml.evaluate import evaluate_saved_model
    print("\n--- Evaluating Saved Pipeline ---")
    return evaluate_saved_model()


def start_backend():
    """Launches the FastAPI backend server."""
    import uvicorn
    print("\n--- Starting LungScan AI FastAPI Backend on http://127.0.0.1:8000 ---")
    print("API Documentation available at: http://127.0.0.1:8000/docs")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)


def main():
    parser = argparse.ArgumentParser(description="LungScan AI - Logistic Regression Lung Tumor Classifier")
    parser.add_argument("--download", action="store_true", help="Download Kaggle dataset via kagglehub")
    parser.add_argument("--train", action="store_true", help="Train Logistic Regression pipeline")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate saved model")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI backend server")
    args = parser.parse_args()

    if args.download:
        download_dataset()
    elif args.train:
        train_model()
    elif args.evaluate:
        evaluate_model()
    elif args.serve:
        start_backend()
    else:
        # Default behavior: verify dataset and start backend
        model_file = os.path.join(BASE_DIR, "backend", "models", "lungscan_lr_pipeline.joblib")
        if not os.path.exists(model_file):
            print("Model not found. Downloading dataset and training pipeline...")
            download_dataset()
            train_model()
        start_backend()


if __name__ == "__main__":
    main()
