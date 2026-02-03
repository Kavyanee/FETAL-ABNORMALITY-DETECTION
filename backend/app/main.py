from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import health, prediction
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Fetal Abnormality Detection API",
    description="Privacy-Preserving AI System for Fetal Ultrasound Analysis",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(prediction.router, tags=["Prediction"])
