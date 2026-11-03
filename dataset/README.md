# IQ-OTH/NCCD Lung Cancer Dataset Documentation

## 1. Overview and Source

The **IQ-OTH/NCCD Lung Cancer Dataset** is a publicly available thoracic CT scan dataset collected at:
- **Iraq-Oncology Teaching Hospital (IQ-OTH)**
- **National Center for Cancer Diseases (NCCD)**

Collected over a period of three months in fall 2019, the dataset includes axial CT scan slices from patients diagnosed with lung cancer at various histological stages, as well as healthy individuals.

- **Kaggle Dataset URL:** https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset
- **Scanner Specification:** Siemens SOMATOM CT
- **Protocol Parameters:** 120 kV tube potential, 1 mm slice thickness, window width 350 to 1200 HU, window center 50 to 600 HU. Full inspiration breath-hold.
- **DICOM De-identification:** All patient personal identifiable information (PII) was stripped prior to public release.

---

## 2. Automated Download via `kagglehub`

To download the latest version of this dataset programmatically in Python:

```python
import kagglehub

# Download latest version of IQ-OTH/NCCD Lung Cancer Dataset
path = kagglehub.dataset_download("hamdallak/the-iqothnccd-lung-cancer-dataset")
print("Path to dataset files:", path)
```

The `backend/ml/load_data.py` module automatically searches the local Kagglehub cache (`~/.cache/kagglehub/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/versions/1`) or triggers this automated download if not yet cached.

---

## 3. Directory Structure

Upon extraction, the dataset has the following folder hierarchy:

```text
The IQ-OTHNCCD lung cancer dataset/
│
├── Bengin cases/                        # 120 files (Note: original spelling 'Bengin')
│   ├── Bengin case (1).jpg
│   └── ...
│
├── Malignant cases/                     # 561 files
│   ├── Malignant case (1).jpg
│   └── ...
│
├── Normal cases/                        # 416 files
│   ├── Normal case (1).jpg
│   └── ...
│
└── IQ-OTH_NCCD lung cancer dataset.txt  # Dataset provenance and protocol notes
```

---

## 4. Binary Tumor Classification Mapping

For the primary task of **Benign vs. Malignant tumor discrimination**:

| Category | Folder Name in Dataset | Class Label | Clinical Significance |
|---|---|---|---|
| **Benign** | `Bengin cases/` | `0` | Non-cancerous pulmonary abnormality (granuloma, hamartoma, scarring) |
| **Malignant** | `Malignant cases/` | `1` | Cancerous pulmonary carcinoma (bronchogenic adenocarcinoma, squamous) |

> **Important Protocol Note:** The 416 `Normal cases` are healthy scans with no nodules. As specified by clinical machine-learning guidelines, normal images are **not** combined with benign lesions because doing so would conflate non-nodular lung tissue with benign pathological abnormalities and inflate classifier accuracy trivially.

---

## 5. Dataset Validation & Deduplication Statistics

When verified through `backend/ml/load_data.py`:
- **Total Valid Binary CT Scans:** 649 unique verified slices
- **Benign Slices (Class 0):** 102 verified slices
- **Malignant Slices (Class 1):** 547 verified slices
- **Excluded Duplicates:** 32 identical duplicate CT slices detected and purged via MD5 hashing
- **Corrupt Files:** 0

---

## 6. Known Dataset Limitations

1. **2D Axial Slices vs 3D Volumetry:** The dataset provides isolated 2D JPEG slices rather than full 3D volumetric DICOM stacks.
2. **Whole-Scan Radiomics:** Without manual or deep nodule contour segmentation masks, handcrafted features reflect both lung parenchyma and thoracic wall boundaries.
3. **Class Imbalance:** Malignant cases outnumber benign cases by ~5.3:1. Class weighting (`class_weight="balanced"`) is therefore mandatory in training.

