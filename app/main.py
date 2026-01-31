"""
FastAPI inference service for Cats vs Dogs classification.
Provides health check, prediction endpoints, and Prometheus metrics.
"""

import os
import sys
import io
import base64
import time
import logging
import requests
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import SimpleCNN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests', ['status'])
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Prediction request latency')
HEALTH_CHECK_COUNT = Counter('health_checks_total', 'Total health check requests')
PREDICTION_BY_CLASS = Counter('prediction_class_total', 'Total predictions by class', ['label'])

# Initialize FastAPI app
app = FastAPI(
    title="End-to-End Cats vs Dogs Classifier",
    description="Binary image classification API for pet adoption platform",
    version="1.0.0"
)

# Global model variable
model = None
model_load_time = None


class PredictionRequest(BaseModel):
    image: Optional[str] = None # Base64 encoded image
    image_url: Optional[str] = None # URL to image

    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?ixlib=rb-4.0.3&auto=format&fit=crop&w=500&q=60"
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    class_id: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    is_model_loaded: bool
    loaded_at: Optional[str]
    timestamp: str


def load_model():
    """Load the trained model."""
    global model, model_load_time
    
    model_paths = [
        'models/cats_dogs_model.npz',
        '/app/models/cats_dogs_model.npz',
        os.path.join(os.path.dirname(__file__), '..', 'models', 'cats_dogs_model.npz')
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            logger.info(f"Loading model from {path}")
            model = SimpleCNN.load(path)
            model_load_time = datetime.now().isoformat()
            logger.info("Model loaded successfully")
            return True
    
    # If no saved model, create a new one (for testing purposes)
    logger.warning("No saved model found, creating new model for testing")
    model = SimpleCNN(input_shape=(224, 224, 3), hidden_units=64)
    model_load_time = datetime.now().isoformat()
    return True


def preprocess_image(image_data: str) -> np.ndarray:
    """Preprocess base64 image for prediction."""
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to expected size
        img = img.resize((224, 224), Image.BILINEAR)
        
        # Convert to numpy array and normalize to [-1, 1]
        img_array = (np.array(img, dtype=np.float32) / 127.5) - 1.0
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ValueError(f"Invalid image data: {e}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    HEALTH_CHECK_COUNT.inc()
    logger.info("Health check requested")
    
    return HealthResponse(
        status="healthy",
        is_model_loaded=model is not None,
        loaded_at=model_load_time,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, req: Request):
    """Prediction endpoint - accepts base64 encoded image."""
    start_time = time.time()
    
    # Log request (without image data for privacy)
    logger.info(f"Prediction request received from {req.client.host}")
    
    if model is None:
        REQUEST_COUNT.labels(status='error').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get image data
        if request.image_url:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(request.image_url, headers=headers, timeout=10)
            response.raise_for_status()
            image_bytes = response.content
            img_array = preprocess_image(base64.b64encode(image_bytes))
        elif request.image:
            img_array = preprocess_image(request.image)
        else:
            raise ValueError("Either 'image' (base64) or 'image_url' must be provided")
        
        # Make prediction
        prob = float(model.predict_proba(img_array)[0])
        class_id = int(prob > 0.5)
        prediction = "dog" if class_id == 1 else "cat"
        confidence = prob if class_id == 1 else 1 - prob
        
        processing_time = (time.time() - start_time) * 1000
        
        # Record metrics
        REQUEST_COUNT.labels(status='success').inc()
        PREDICTION_BY_CLASS.labels(label=prediction).inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        
        # Log response
        logger.info(f"Prediction: {prediction} (confidence: {confidence:.4f}, time: {processing_time:.2f}ms)")
        
        return PredictionResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            class_id=class_id,
            processing_time_ms=round(processing_time, 2)
        )
        
    except ValueError as e:
        REQUEST_COUNT.labels(status='error').inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        REQUEST_COUNT.labels(status='error').inc()
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Cats vs Dogs Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
