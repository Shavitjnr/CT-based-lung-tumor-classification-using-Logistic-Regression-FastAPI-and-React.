"""
main.py - LungScan AI FastAPI Application Entrypoint
AI-Assisted Lung Tumor Classification Using Logistic Regression and CT Scan Analysis
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.model_info import router as model_info_router
from backend.routes.predict import router as predict_router

app = FastAPI(
    title="LungScan AI - Medical CT Scan Classification API",
    description="Educational radiomic classification backend utilizing handcrafted features and Logistic Regression on the IQ-OTH/NCCD dataset.",
    version="1.0.0"
)

# Enable CORS for local Vite development and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(model_info_router)
app.include_router(predict_router)

# Mount samples folder for direct static access if needed
samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
if os.path.exists(samples_path):
    app.mount("/static/samples", StaticFiles(directory=samples_path), name="samples")


@app.get("/")
def root_endpoint():
    return {
        "project": "LungScan AI",
        "tagline": "AI-Assisted Lung Tumor Classification",
        "methodology": "Handcrafted Radiomics + Logistic Regression",
        "documentation": "/docs",
        "endpoints": [
            "/api/health",
            "/api/model/metrics",
            "/api/model/info",
            "/api/classes",
            "/api/samples",
            "/api/predict"
        ],
        "disclaimer": "Academic and educational research prototype only. Not for clinical diagnostic use."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

