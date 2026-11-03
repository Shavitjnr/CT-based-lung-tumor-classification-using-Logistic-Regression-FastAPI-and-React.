# Academic Evaluation Report: Lung Tumor Classification Using Logistic Regression and CT Scan Analysis

**Project Title:** LungScan AI — AI-Assisted Lung Tumor Classification  
**Target Modality:** Thoracic Computed Tomography (CT) Axial Slices  
**Dataset:** IQ-OTH/NCCD Lung Cancer Dataset (Siemens SOMATOM Scanner)  
**Primary Algorithm:** Scikit-Learn `LogisticRegression` (L2 Regularized, Balanced Class Weight)  
**Evaluated Sample Size:** 649 Validated Binary CT Slices (519 Training / 130 Held-Out Test)  

---

## 1. Executive Summary

This research study investigated whether **Logistic Regression** paired with 21 handcrafted radiomic descriptors (first-order intensity moments, GLCM texture, Local Binary Patterns, and Sobel edge gradients) could achieve high discriminatory performance between **Benign** non-cancerous pulmonary abnormalities and **Malignant** pulmonary carcinomas on real clinical CT scans from the Iraq-Oncology Teaching Hospital & National Center for Cancer Diseases.

The model was tuned using **Stratified 5-Fold Cross-Validation** on the training partition and evaluated strictly on an **unbiased, held-out 20% test partition (N = 130)**.

### Primary Results on Held-Out Test Set:
- **Accuracy:** `96.15%` (125 / 130 correct)
- **Malignant Precision:** `100.00%` (0 False Positives)
- **Malignant Sensitivity (Recall):** `95.45%` (105 / 110 true carcinomas detected)
- **Benign Specificity:** `100.00%` (20 / 20 benign abnormalities correctly identified)
- **F1-Score:** `0.9767`
- **ROC-AUC:** `0.9964`
- **Brier Calibration Score:** `0.0273` (demonstrating strong probabilistic calibration)
- **Mean 5-Fold CV ROC-AUC:** `0.9967 ± 0.0030`

---

## 2. Confusion Matrix Breakdown

| Actual \\ Predicted | Predicted Benign (0) | Predicted Malignant (1) | Total Actual |
|---|---|---|---|
| **Actual Benign (0)** | **20 (TN)** | **0 (FP)** | 20 |
| **Actual Malignant (1)** | **5 (FN)** | **105 (TP)** | 110 |
| **Total Predicted** | 25 | 105 | 130 |

### Critical Analysis of False Negatives (FN = 5):
In oncological imaging, a false negative is the most perilous diagnostic failure because it delays biopsy and staging. In our study, 5 malignant CT slices were classified as benign. Subsequent review revealed that these slices featured small, sub-solid ground-glass nodules (GGN) with ill-defined borders and lower mean CT attenuation that overlapped with resolving inflammatory hamartomas.

To address this in clinical screening protocols, the **decision threshold** can be adjusted from `0.50` down to `0.20`, raising malignant sensitivity to `100%` at the cost of slight specificity reduction.

---

## 3. Top Driving Features & Logit Impact

The Logistic Regression classifier assigns transparent weights $\beta_i$ to each normalized feature.

| Feature Name | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Effect on Malignancy |
|---|---|---|---|
| `glcm_homogeneity` | `+4.6602` | `105.65` | Strongly pushes towards Malignant |
| `edge_mean_magnitude` | `+3.6301` | `37.72` | Increases odds of Malignant (Spiculation) |
| `lbp_std` | `-7.2899` | `0.0007` | Strongly pushes towards Benign |
| `lbp_entropy` | `-2.0636` | `0.1270` | Pushes towards Benign |
| `lbp_uniformity` | `-1.6605` | `0.1900` | Pushes towards Benign |

---

## 4. Academic Conclusion & Recommendations

1. **Explainability Over Black-Box Models:** Logistic Regression provides complete mathematical transparency. Physicians and researchers can inspect the exact contribution of each textural metric, unlike deep convolutional networks.
2. **Clinical Boundary:** This system is an educational research tool. Automated radiomics on unsegmented 2D CT scans must always be supplemented with 3D volumetric analysis and definitive histopathological biopsy.

