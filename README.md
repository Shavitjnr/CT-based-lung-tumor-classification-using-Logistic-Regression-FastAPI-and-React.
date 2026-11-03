# LungScan AI: Lung Tumor Classification Using Logistic Regression & CT Scan Analysis

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%200.128-009688.svg?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%201.6-F7931E.svg?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![React](https://img.shields.io/badge/Frontend-React%2019%20%2B%20TypeScript-61DAFB.svg?style=flat-square&logo=react)](https://react.dev)
[![Tailwind CSS](https://img.shields.io/badge/UI-Tailwind%20CSS%204.0%20(Sharp%20Edges)-38B2AC.svg?style=flat-square&logo=tailwind-css)](https://tailwindcss.com)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-IQ--OTH%2FNCCD%20CT%20Scans-20BEFF.svg?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)

An academic, end-to-end medical machine learning system and full-stack clinical research web application developed for **B.Tech Computer Science (Artificial Intelligence & Machine Learning)**.

The system analyzes thoracic Computed Tomography (CT) scans to differentiate between **Benign (non-cancerous)** abnormalities and **Malignant (cancerous)** pulmonary carcinomas using **handcrafted radiomic feature extraction** and a regularized **Logistic Regression** classifier.

---

## 1. Project Overview & Clinical Rationale

In clinical thoracic oncology, early distinction between non-cancerous benign lesions (e.g., granulomas, hamartomas, post-inflammatory scarring) and primary malignant carcinomas is vital. While deep convolutional neural networks (CNNs) often perform as opaque "black boxes," **Logistic Regression operating on handcrafted radiomic features** provides complete mathematical transparency:

$$\text{logit}(P) = \beta_0 + \sum_{i=1}^{21} \beta_i \cdot Z_i$$

$$P(\text{Malignant} \mid X) = \frac{1}{1 + e^{-\text{logit}}}$$

Every prediction can be decomposed into exact log-odds contributions, allowing radiologists and researchers to understand *why* the model made its decision based on quantifiable physical descriptors (attenuation heterogeneity, GLCM texture, Local Binary Patterns, and edge spiculation).

> **Academic & Medical Notice:** This application is an educational and research prototype. It does not constitute a certified medical diagnostic device and must not be used to make treatment decisions.

---

## 2. Dataset Information

The model is trained and evaluated on the verified **IQ-OTH/NCCD Lung Cancer Dataset**:
- **Source:** Iraq-Oncology Teaching Hospital & National Center for Cancer Diseases (Fall 2019)
- **Scanner:** Siemens SOMATOM CT (120 kV tube potential, 1 mm slice thickness, 350-1200 HU window)
- **URL:** [https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)
- **Target Task:** Binary Classification (`Benign = 0`, `Malignant = 1`)
- **Deduplication:** 32 duplicate CT slices identified via MD5 hash and purged
- **Clean Sample Count:** 649 unique verified slices (102 Benign, 547 Malignant; 416 normal scans documented and excluded from binary tumor task to avoid trivial background bias)

---

## 3. Real Model Evaluation Results

Evaluated on an **unbiased held-out test split (20%, N = 130 verified CT slices)** following Stratified 5-Fold Cross-Validation:

| Metric | Score | Clinical Interpretation |
|---|---|---|
| **Accuracy** | **96.15%** | 125 of 130 test slices correctly classified |
| **ROC-AUC** | **0.9964** | Near-perfect class separability |
| **Malignant Precision** | **100.00%** | Zero False Positives (0 benign cases mislabeled as cancer) |
| **Malignant Recall (Sensitivity)** | **95.45%** | 105 of 110 malignant carcinomas detected |
| **Benign Specificity** | **100.00%** | 20 of 20 benign lesions correctly verified |
| **F1-Score** | **0.9767** | Harmonic mean of precision and recall |
| **Brier Score** | **0.0273** | Well-calibrated probabilistic output |
| **5-Fold CV Mean ROC-AUC** | **0.9967 ± 0.003** | Robust generalization across training folds |

### Confusion Matrix on Held-Out Test Set (N = 130):
```text
                     Predicted Benign (0)    Predicted Malignant (1)
Actual Benign (0)            20 (TN)                  0 (FP)
Actual Malignant (1)          5 (FN)                105 (TP)
```

*False Negative Discussion:* The 5 false negatives represent subtle, ground-glass opacity (GGO) sub-centimeter nodules with lower CT attenuation. The web dashboard includes an interactive threshold slider allowing clinicians to shift the decision boundary (e.g., to 0.20) for maximum sensitivity screening.

---

## 4. Radiomic Feature Engineering (21 Features)

The pipeline extracts 21 mathematical descriptors across four families:

1. **First-Order Intensity Moments:** `mean_intensity`, `std_intensity`, `skewness`, `kurtosis`, `shannon_entropy`, `percentile_25`, `percentile_75`, `iqr_intensity`.
2. **GLCM Texture Metrics (Gray-Level Co-occurrence Matrix):** `glcm_contrast`, `glcm_dissimilarity`, `glcm_homogeneity`, `glcm_energy`, `glcm_correlation`, `glcm_asm`.
3. **Local Binary Patterns (LBP):** `lbp_mean`, `lbp_std`, `lbp_entropy`, `lbp_uniformity`.
4. **Boundary & Morphological Edge Metrics:** `edge_density`, `edge_mean_magnitude`, `parenchyma_attenuation_ratio`.

---

## 5. Technology Stack

- **Machine Learning & Core:** Python 3.9+, NumPy, Pandas, Scikit-Learn 1.6, Scikit-Image, OpenCV, SciPy, Joblib, KaggleHub
- **Backend API:** FastAPI, Uvicorn, Pydantic, Python-Multipart, Pytest
- **Frontend Dashboard:** React 19, TypeScript, Vite, Tailwind CSS 4.0, Lucide React, Recharts
- **UI Design System:** Crisp, sharp clinical interface (zero blurry glow / no flaskbox effect, clean dark navy `#0f172a` & white palette with `#0d9488` teal healthcare accents, monospace telemetry typography).

---

## 6. Project Structure

```text
Cancer Pridiction/
├── backend/
│   ├── main.py                     # FastAPI server & route orchestration
│   ├── requirements.txt            # Python dependencies
│   ├── routes/
│   │   ├── predict.py              # POST /api/predict (CT upload & inference)
│   │   └── model_info.py           # GET /api/model/metrics, /api/model/info, /api/samples
│   ├── ml/
│   │   ├── load_data.py            # Kaggle dataset scanner, validator & deduplicator
│   │   ├── preprocess.py           # CLAHE, Gaussian denoise, 256x256 resizing
│   │   ├── feature_extraction.py   # 21 GLCM, LBP, Intensity, and Edge features
│   │   ├── train.py                # Stratified 5-Fold CV, GridSearch, pipeline training
│   │   ├── evaluate.py             # Standalone test evaluation & classification report
│   │   └── predict.py              # Real-time inference & logit explainability
│   ├── models/
│   │   ├── lungscan_lr_pipeline.joblib   # Trained StandardScaler + LogisticRegression
│   │   └── model_metadata.json           # Real evaluation metrics & feature weights
│   ├── samples/                    # Curated verified sample CT scans for instant demo
│   └── tests/
│       └── test_api.py             # 8 automated pytest integration tests
│
├── frontend/
│   ├── src/
│   │   ├── components/Navbar.tsx   # Sharp clinical navigation bar
│   │   ├── pages/
│   │   │   ├── HomeDashboard.tsx   # Landing page with study highlights & metrics
│   │   │   ├── CTScanUpload.tsx    # Drag-and-drop upload & 1-click sample gallery
│   │   │   ├── PredictionResults.tsx # Class breakdown, logit decomposition & radar table
│   │   │   ├── ModelPerformance.tsx  # Interactive ROC & PR curves, confusion matrix
│   │   │   └── AboutProject.tsx    # System flowchart, mathematical formulation
│   │   ├── services/api.ts         # Typed API client services
│   │   ├── App.tsx                 # Root application state & router
│   │   ├── main.tsx                # React DOM mount
│   │   └── index.css               # Sharp edge styling & Tailwind directives
│   ├── package.json
│   └── vite.config.ts              # Vite config with backend proxy
│
├── dataset/
│   └── README.md                   # Full IQ-OTH/NCCD dataset documentation
├── notebooks/
│   └── experimentation.ipynb       # Jupyter exploratory analysis notebook
├── reports/
│   └── model_evaluation_report.md  # Formal academic evaluation report
├── main.py                         # Root CLI launcher
└── README.md
```

---

## 7. Installation & Quick Start

### Step 1: Clone and Set Up Python Virtual Environment

```bash
# Navigate to the repository
cd "Cancer Pridiction"

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install ML and backend dependencies
pip install -r backend/requirements.txt
```

### Step 2: Download Dataset & Train Model

```bash
# Download dataset via kagglehub and run complete training pipeline
python main.py --train

# (Optional) Run automated tests
pytest backend/tests/test_api.py
```

### Step 3: Run Backend Server

```bash
# Launch FastAPI backend
python main.py --serve
# Backend will be active on http://127.0.0.1:8000
# Interactive Swagger docs: http://127.0.0.1:8000/docs
```

### Step 4: Run Frontend Dashboard

In a new terminal window:

```bash
cd frontend

# Install Node dependencies
npm install

# Start Vite development server
npm run dev
# Frontend will be live on http://127.0.0.1:5173
```

Open **http://127.0.0.1:5173** in your browser.

---

## 8. Backend API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health and model loading status |
| `POST` | `/api/predict` | Upload CT scan or pass `sample_id` to receive classification, probabilities, and logit feature contributions |
| `GET` | `/api/model/metrics` | Returns held-out test evaluation metrics, confusion matrix, ROC points, and PR points |
| `GET` | `/api/model/info` | Returns dataset metadata, feature descriptions, and model coefficients |
| `GET` | `/api/classes` | Returns target class definitions (Benign: 0, Malignant: 1) |
| `GET` | `/api/samples` | Curated sample CT scans for 1-click immediate demonstration |
| `GET` | `/api/samples/image/{filename}` | Serves verified sample CT scan images |

---

## 9. Academic Project Deliverables Checklist

- [x] Automated Kaggle dataset download via `kagglehub`
- [x] Comprehensive reproducible preprocessing pipeline (CLAHE, Gaussian denoise, uniform resize)
- [x] Handcrafted radiomics extraction (21 statistical, GLCM, LBP, and edge features)
- [x] StandardScaler + Logistic Regression pipeline with L1/L2 hyperparameter tuning
- [x] Stratified 5-Fold Cross-Validation on training data and unbiased final test set evaluation
- [x] True confusion matrix, ROC curve, and Precision-Recall curve data
- [x] Saved model pipeline (`lungscan_lr_pipeline.joblib`) and metadata (`model_metadata.json`)
- [x] Robust FastAPI backend with input validation and automated tests (`pytest`)
- [x] React + TypeScript + Vite + Tailwind frontend with sharp edges and premium clinical styling
- [x] 1-Click test case gallery featuring real CT scans for instant verification
- [x] Formal academic evaluation report and technical documentation

