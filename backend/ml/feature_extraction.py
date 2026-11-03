"""
feature_extraction.py - Handcrafted Radiomic Feature Extraction for Lung CT Scans
Part of LungScan AI Project.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2


FEATURE_DESCRIPTIONS = {
    "mean_intensity": "Average CT attenuation/grayscale intensity across the scan",
    "std_intensity": "Standard deviation of CT attenuation reflecting tissue heterogeneity",
    "skewness": "Asymmetry of pixel intensity distribution (positive = bright lesion dominance)",
    "kurtosis": "Peakedness and heavy tails of CT attenuation histogram",
    "shannon_entropy": "Shannon information entropy measuring gray-level randomness and complexity",
    "percentile_25": "25th percentile of normalized CT attenuation",
    "percentile_75": "75th percentile of normalized CT attenuation",
    "iqr_intensity": "Interquartile range (IQR) of CT intensity spread",
    "glcm_contrast": "GLCM Contrast measuring local intensity variations between voxel pairs",
    "glcm_dissimilarity": "GLCM Dissimilarity quantifying average difference between adjacent pixels",
    "glcm_homogeneity": "GLCM Homogeneity indicating structural smoothness and uniformity",
    "glcm_energy": "GLCM Energy measuring texture uniformity and orderliness",
    "glcm_correlation": "GLCM Correlation measuring linear spatial dependencies between pixels",
    "glcm_asm": "GLCM Angular Second Moment reflecting pixel pair regularity",
    "lbp_mean": "Mean value of uniform Local Binary Patterns (micro-texture)",
    "lbp_std": "Standard deviation of Local Binary Patterns representing surface roughness",
    "lbp_entropy": "Shannon entropy of local binary pattern code distribution",
    "lbp_uniformity": "Proportion of uniform micro-patterns across the scan",
    "edge_density": "Fraction of sharp edge pixels representing pleural tags or lesion boundaries",
    "edge_mean_magnitude": "Mean Sobel gradient magnitude reflecting boundary sharpness/spiculation",
    "parenchyma_attenuation_ratio": "Proportion of high-density tissue within the segmented lung fields",
}

ORDERED_FEATURE_NAMES = list(FEATURE_DESCRIPTIONS.keys())


def compute_intensity_stats(gray: np.ndarray) -> Dict[str, float]:
    """Computes first-order statistical moments and quantile metrics."""
    # Scale to [0, 1] for numerical stability
    norm = gray.astype(np.float32) / 255.0
    flat = norm.flatten()

    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    skew_val = float(stats.skew(flat))
    kurt_val = float(stats.kurtosis(flat))

    p25, p75 = np.percentile(flat, [25, 75])
    iqr_val = float(p75 - p25)

    # Shannon Entropy using 64-bin histogram
    hist, _ = np.histogram(gray, bins=64, range=(0, 256), density=True)
    hist = hist[hist > 0]
    shannon_entropy = float(-np.sum(hist * np.log2(hist)))

    return {
        "mean_intensity": round(mean_val, 5),
        "std_intensity": round(std_val, 5),
        "skewness": round(skew_val, 5),
        "kurtosis": round(kurt_val, 5),
        "shannon_entropy": round(shannon_entropy, 5),
        "percentile_25": round(float(p25), 5),
        "percentile_75": round(float(p75), 5),
        "iqr_intensity": round(iqr_val, 5),
    }


def compute_glcm_features(gray: np.ndarray) -> Dict[str, float]:
    """
    Computes Gray-Level Co-occurrence Matrix (GLCM) second-order texture features
    across distances=[1, 2] and angles=[0, 45, 90, 135 degrees].
    Quantizes to 32 gray levels for robust, invariant texture representation.
    """
    # Quantize 256 levels to 32 levels
    quantized = (gray // 8).astype(np.uint8)

    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        quantized,
        distances=distances,
        angles=angles,
        levels=32,
        symmetric=True,
        normed=True
    )

    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    dissimilarity = float(np.mean(graycoprops(glcm, "dissimilarity")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))
    asm = float(np.mean(graycoprops(glcm, "ASM")))

    return {
        "glcm_contrast": round(contrast, 5),
        "glcm_dissimilarity": round(dissimilarity, 5),
        "glcm_homogeneity": round(homogeneity, 5),
        "glcm_energy": round(energy, 5),
        "glcm_correlation": round(correlation, 5),
        "glcm_asm": round(asm, 5),
    }


def compute_lbp_features(gray: np.ndarray) -> Dict[str, float]:
    """
    Computes Local Binary Patterns (LBP) to capture micro-texture and fine spatial structures.
    Uses circular uniform LBP with radius=1 and n_points=8.
    """
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

    lbp_mean = float(np.mean(lbp))
    lbp_std = float(np.std(lbp))

    # LBP Histogram (uniform method produces n_points + 2 bins: 10 bins)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist_nonzero = hist[hist > 0]
    lbp_entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))
    lbp_uniformity = float(np.sum(hist ** 2))

    return {
        "lbp_mean": round(lbp_mean, 5),
        "lbp_std": round(lbp_std, 5),
        "lbp_entropy": round(lbp_entropy, 5),
        "lbp_uniformity": round(lbp_uniformity, 5),
    }


def compute_edge_features(gray: np.ndarray) -> Dict[str, float]:
    """
    Computes edge density and mean gradient magnitude using Sobel operators.
    Captures lesion boundaries, spiculation, and vascular markings.
    """
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Edge threshold at 10% of max theoretical gradient
    edge_thresh = 50.0
    edge_pixels = np.count_nonzero(magnitude > edge_thresh)
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_density = float(edge_pixels) / float(total_pixels)
    edge_mean_magnitude = float(np.mean(magnitude))

    return {
        "edge_density": round(edge_density, 5),
        "edge_mean_magnitude": round(edge_mean_magnitude, 5),
    }


def compute_parenchyma_features(gray: np.ndarray, lung_mask: np.ndarray) -> Dict[str, float]:
    """
    Measures non-air high attenuation ratio within the segmented lung fields.
    Malignant tumors introduce solid/subsolid masses with attenuation higher than aerated lung.
    """
    if lung_mask is None or np.count_nonzero(lung_mask) == 0:
        return {"parenchyma_attenuation_ratio": 0.0}

    parenchyma_pixels = gray[lung_mask > 0]
    if len(parenchyma_pixels) == 0:
        return {"parenchyma_attenuation_ratio": 0.0}

    # Threshold for solid/subsolid tissue vs aerated lung (> 80 in 8-bit scale)
    high_att_count = np.count_nonzero(parenchyma_pixels > 80)
    ratio = float(high_att_count) / float(len(parenchyma_pixels))
    return {"parenchyma_attenuation_ratio": round(ratio, 5)}


def extract_features(preprocessed_dict: Dict[str, any]) -> Dict[str, float]:
    """
    Extracts all handcrafted features from a preprocessed scan.
    Returns ordered dictionary of numerical radiomic features.
    """
    gray = preprocessed_dict["processed_gray"]
    mask = preprocessed_dict.get("lung_mask")

    features = {}
    features.update(compute_intensity_stats(gray))
    features.update(compute_glcm_features(gray))
    features.update(compute_lbp_features(gray))
    features.update(compute_edge_features(gray))
    features.update(compute_parenchyma_features(gray, mask))

    return {k: features.get(k, 0.0) for k in ORDERED_FEATURE_NAMES}


def extract_feature_vector(preprocessed_dict: Dict[str, any]) -> np.ndarray:
    """Extracts features as a 1D numpy array in standard feature order."""
    feat_dict = extract_features(preprocessed_dict)
    return np.array([feat_dict[k] for k in ORDERED_FEATURE_NAMES], dtype=np.float32)


def get_feature_names() -> List[str]:
    """Returns list of feature names in fixed order."""
    return list(ORDERED_FEATURE_NAMES)


def get_feature_metadata() -> Dict[str, str]:
    """Returns feature documentation dictionary."""
    return dict(FEATURE_DESCRIPTIONS)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from backend.ml.preprocess import preprocess_pipeline
    dummy = np.random.randint(40, 220, (256, 256, 3), dtype=np.uint8)
    pre = preprocess_pipeline(dummy)
    feats = extract_features(pre)
    print("Extracted Features count:", len(feats))
    for k, v in list(feats.items())[:6]:
        print(f"  {k}: {v} ({FEATURE_DESCRIPTIONS[k]})")
