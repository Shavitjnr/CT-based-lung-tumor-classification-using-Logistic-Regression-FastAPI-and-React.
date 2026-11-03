"""
preprocess.py - Medical Image Preprocessing Pipeline for Thoracic CT Scans
Part of LungScan AI Project.
"""

from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
import cv2


TARGET_SIZE = (256, 256)


def load_raw_image(image_input: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """
    Loads raw image from a file path, in-memory bytes, or existing numpy array.
    Returns BGR or Grayscale numpy array.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Unable to read image from path: {image_input}")
        return img
    elif isinstance(image_input, (bytes, bytearray)):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image from provided byte stream.")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Converts image to single-channel 8-bit grayscale.
    """
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unexpected image shape for grayscale conversion: {img.shape}")


def resize_image(gray: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Resizes image to consistent target dimension using area interpolation for downsampling
    or cubic interpolation for upsampling.
    """
    h, w = gray.shape[:2]
    if (w, h) == target_size:
        return gray
    interp = cv2.INTER_AREA if (w > target_size[0] or h > target_size[1]) else cv2.INTER_CUBIC
    return cv2.resize(gray, target_size, interpolation=interp)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance lung nodule contrast and soft tissue density differentiation.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def denoise_image(gray: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Applies gentle Gaussian smoothing to reduce thermal/quantum CT scanner noise
    while preserving spiculation and margins of lung lesions.
    """
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0.5)


def extract_lung_parenchyma_mask(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes an approximate lung cavity mask using Otsu thresholding and morphological closing.
    Returns:
        mask (np.ndarray): Binary mask where 255 indicates thoracic cavity / lung fields.
        parenchyma_ratio (float): Ratio of lung cavity area to total image area.
    """
    # In chest CT, aerated lung parenchyma is low attenuation (darker), surrounding chest wall is brighter
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    total_pixels = gray.shape[0] * gray.shape[1]
    mask_pixels = np.count_nonzero(cleaned)
    ratio = float(mask_pixels) / float(total_pixels)
    return cleaned, ratio


def preprocess_pipeline(
    image_input: Union[str, bytes, np.ndarray],
    target_size: Tuple[int, int] = TARGET_SIZE
) -> Dict[str, Any]:
    """
    Full reproducible preprocessing pipeline applied consistently across training,
    testing, and real-time user uploads.
    """
    raw = load_raw_image(image_input)
    gray = to_grayscale(raw)
    resized = resize_image(gray, target_size=target_size)
    denoised = denoise_image(resized)
    enhanced = apply_clahe(denoised)
    mask, parenchyma_ratio = extract_lung_parenchyma_mask(enhanced)
    
    # Normalized float representation [0.0, 1.0]
    normalized_float = enhanced.astype(np.float32) / 255.0

    return {
        "raw_shape": raw.shape,
        "processed_gray": enhanced,
        "normalized_float": normalized_float,
        "lung_mask": mask,
        "parenchyma_ratio": parenchyma_ratio,
        "target_size": target_size,
    }


if __name__ == "__main__":
    # Smoke test on dummy image
    dummy = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    res = preprocess_pipeline(dummy)
    print("Preprocessing smoke test successful:", {k: (v.shape if hasattr(v, 'shape') else v) for k, v in res.items()})
