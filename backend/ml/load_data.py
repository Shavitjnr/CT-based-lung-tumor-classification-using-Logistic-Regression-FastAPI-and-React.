"""
load_data.py - IQ-OTH/NCCD Lung Cancer Dataset Loader and Validator
Part of LungScan AI Project.
"""

import os
import glob
import hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import kagglehub


def get_dataset_root(custom_path: Optional[str] = None) -> str:
    """
    Locates the IQ-OTH/NCCD Lung Cancer Dataset.
    Checks custom path, cached kagglehub download, or triggers download.
    """
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # Check local cache path
    cache_path = os.path.expanduser("~/.cache/kagglehub/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset/versions/1")
    if os.path.exists(cache_path):
        sub = os.path.join(cache_path, "The IQ-OTHNCCD lung cancer dataset")
        if os.path.exists(sub):
            return sub
        return cache_path

    # Download via kagglehub if not found
    print("Dataset not found locally. Downloading via kagglehub...")
    download_dir = kagglehub.dataset_download("hamdallak/the-iqothnccd-lung-cancer-dataset")
    sub = os.path.join(download_dir, "The IQ-OTHNCCD lung cancer dataset")
    if os.path.exists(sub):
        return sub
    return download_dir


def compute_file_hash(filepath: str) -> str:
    """Computes MD5 hash for image deduplication check."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def scan_dataset(dataset_dir: str) -> Dict[str, any]:
    """
    Scans the IQ-OTH/NCCD dataset directory, detects class folders,
    validates images, removes duplicates, and records statistics.

    Binary Classification Mapping:
    - Benign (Bengin cases): Class 0
    - Malignant (Malignant cases): Class 1
    (Normal cases are excluded from primary binary tumor classification per specification).
    """
    print(f"Scanning dataset at: {dataset_dir}")
    
    # Map folder names (including dataset typo 'Bengin')
    folder_mapping = {
        "Bengin cases": ("Benign", 0),
        "Benign cases": ("Benign", 0),
        "Malignant cases": ("Malignant", 1),
    }

    supported_extensions = {".jpg", ".jpeg", ".png"}
    records = []
    seen_hashes = {}
    corrupted_files = []
    duplicates = []
    normal_case_count = 0

    # Count normal cases for documentation
    normal_dir = os.path.join(dataset_dir, "Normal cases")
    if os.path.exists(normal_dir):
        normal_files = [f for f in os.listdir(normal_dir) if os.path.splitext(f.lower())[1] in supported_extensions]
        normal_case_count = len(normal_files)

    for folder_name, (class_name, class_label) in folder_mapping.items():
        folder_path = os.path.join(dataset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            ext = os.path.splitext(filename.lower())[1]
            if ext not in supported_extensions:
                continue

            filepath = os.path.join(folder_path, filename)
            
            # Check duplicate hash
            try:
                f_hash = compute_file_hash(filepath)
                if f_hash in seen_hashes:
                    duplicates.append((filepath, seen_hashes[f_hash]))
                    continue
                seen_hashes[f_hash] = filepath
            except Exception as e:
                corrupted_files.append((filepath, f"Hash error: {e}"))
                continue

            # Validate image readable
            try:
                img = cv2.imread(filepath)
                if img is None or img.size == 0:
                    corrupted_files.append((filepath, "Unreadable or empty image"))
                    continue
                h, w, c = img.shape
            except Exception as e:
                corrupted_files.append((filepath, f"OpenCV read error: {e}"))
                continue

            records.append({
                "filepath": filepath,
                "filename": filename,
                "class_name": class_name,
                "class_label": class_label,
                "width": w,
                "height": h,
                "channels": c,
                "file_size_bytes": os.path.getsize(filepath)
            })

    # Summary
    benign_count = sum(1 for r in records if r["class_label"] == 0)
    malignant_count = sum(1 for r in records if r["class_label"] == 1)

    print(f"\n--- Dataset Loading Summary ---")
    print(f"Total Valid Binary Records: {len(records)}")
    print(f"  - Benign (Class 0):    {benign_count}")
    print(f"  - Malignant (Class 1): {malignant_count}")
    print(f"  - Normal (Excluded):   {normal_case_count}")
    print(f"Duplicates Excluded:     {len(duplicates)}")
    print(f"Corrupt Files:           {len(corrupted_files)}")

    return {
        "dataset_dir": dataset_dir,
        "records": records,
        "benign_count": benign_count,
        "malignant_count": malignant_count,
        "normal_count": normal_case_count,
        "duplicates_count": len(duplicates),
        "corrupted_count": len(corrupted_files),
        "corrupted_details": corrupted_files,
    }


if __name__ == "__main__":
    root = get_dataset_root()
    summary = scan_dataset(root)

