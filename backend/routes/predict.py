"""
predict.py - FastAPI route for CT scan prediction and radiomic feature analysis.
Part of LungScan AI Project.
"""

import os
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from backend.ml.predict import predict_ct_scan

router = APIRouter(prefix="/api", tags=["Prediction"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
samples_dir = os.path.join(backend_dir, "samples")


@router.post("/predict")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    threshold: Optional[float] = Form(0.50)
):
    """
    Accepts an uploaded chest CT scan image or pre-loaded sample ID,
    extracts radiomic features, and predicts benign vs malignant classification.
    """
    image_bytes = None

    if sample_id:
        # Load pre-curated sample image
        sample_path = os.path.join(samples_dir, f"sample_{sample_id}.jpg")
        if not os.path.exists(sample_path):
            raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        with open(sample_path, "rb") as f:
            image_bytes = f.read()

    elif file is not None:
        filename = file.filename or ""
        ext = os.path.splitext(filename.lower())[1]
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Please upload JPG, JPEG, or PNG thoracic CT scan."
            )

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds maximum permissible size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
            )
        image_bytes = content

    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded CT image or sample_id.")

    try:
        result = predict_ct_scan(image_bytes, threshold=threshold)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image processing and inference failed: {str(e)}"
        )

