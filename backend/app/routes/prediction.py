"""
Prediction API Routes

This module defines the prediction endpoint for ultrasound image analysis.

Endpoints:
- POST /predict: Upload image and get prediction
- GET /model-info: Get model information
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import sys
import os
import traceback
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.ml_service import get_ml_service


router = APIRouter()


class PredictionResponse(BaseModel):
    """
    Response model for prediction endpoint
    """
    prediction: str
    confidence: float
    probabilities: dict
    message: str


class ModelInfoResponse(BaseModel):
    """
    Response model for model info endpoint
    """
    status: str
    model_path: Optional[str] = None
    input_shape: Optional[str] = None
    classes: Optional[list] = None
    total_parameters: Optional[int] = None


@router.post("/predict", response_model=PredictionResponse)
async def predict_abnormality(file: UploadFile = File(...)):
    """
    Predict fetal abnormality from ultrasound image
    
    Args:
        file: Uploaded ultrasound image (JPEG, PNG, etc.)
    
    Returns:
        Prediction result with confidence scores
    
    Example:
        curl -X POST "http://localhost:8000/predict" \
             -H "accept: application/json" \
             -H "Content-Type: multipart/form-data" \
             -F "file=@ultrasound.jpg"
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Get ML service
        ml_service = get_ml_service()
        
        # Run prediction
        result = ml_service.predict(image_bytes)
        
        # Add message based on prediction
        if result['prediction'] == 'Normal':
            message = "No abnormalities detected. Fetus appears normal."
        elif result['prediction'] == 'Benign':
            message = "Benign condition detected. Consult with specialist for monitoring."
        else:  # Malignant
            message = "Malignant condition detected. Immediate specialist consultation required."
        
        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message=message
        )
    
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about the loaded ML model
    
    Returns:
        Model metadata and status
    
    Example:
        curl -X GET "http://localhost:8000/model-info"
    """
    try:
        ml_service = get_ml_service()
        info = ml_service.get_model_info()
        
        return ModelInfoResponse(**info)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Fetal Abnormality Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload ultrasound image for prediction",
            "GET /model-info": "Get model information",
            "GET /health": "Health check"
        },
        "privacy": "This system uses Federated Learning - no patient data is shared",
        "documentation": "/docs"
    }
