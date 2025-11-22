# main.py
import sys
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import numpy as np
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any
from pydantic import BaseModel

# Add root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import settings
from real_detection import real_detector, HAS_ML_MODELS

# Database setup with SQLAlchemy
engine = create_engine(settings.db.url, echo=settings.db.echo)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Pydantic models for responses
class DetectResponse(BaseModel):
    status: str
    detection_type: str
    people_detected: int
    anomalies_detected: int
    anomalies: List[Dict]
    people_data: List[Dict]
    ml_models: bool
    calibration_status: Dict
    message: str

class AnomalyEvent(BaseModel):
    id: int
    track_id: int
    anomaly_type: str
    confidence: float
    description: str
    timestamp: str

class AnomaliesResponse(BaseModel):
    total: int
    anomalies: List[AnomalyEvent]

class HealthResponse(BaseModel):
    status: str
    service: str
    timestamp: str
    ml_models: bool
    calibration_status: Dict

class StatusResponse(BaseModel):
    system: str
    ml_models: bool
    calibration: Dict
    timestamp: str

# App
app = FastAPI(
    title="Biometric Anomaly Detection System",
    description="AI system for detecting anomalies in biometric data with ML models",
    version="2.0.0"
)

# CORS (restricted for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost"],  # Add your frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_database() -> bool:
    """Initialize database"""
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS anomaly_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    anomaly_type TEXT,
                    confidence REAL,
                    description TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_timestamp ON anomaly_events(timestamp)"))
        logger.info("Database initialized")
        return True
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Anomaly Detection System")
    logger.info(f"ML Models: {'ENABLED' if HAS_ML_MODELS else 'DISABLED'}")
    if init_database():
        logger.info("System ready")
    else:
        logger.error("System initialization failed")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down system")

@app.get("/", response_model=Dict[str, Any])
async def root():
    return {
        "message": "Biometric Anomaly Detection System",
        "status": "running",
        "version": "2.0.0",
        "detection_type": "REAL_YOLO",
        "ml_models": HAS_ML_MODELS
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "service": "anomaly_detection",
        "timestamp": datetime.now().isoformat(),
        "ml_models": HAS_ML_MODELS,
        "calibration_status": real_detector.get_calibration_status()
    }

@app.post("/detect", response_model=DetectResponse)
async def detect_anomaly(file: UploadFile = File(...)):
    """Detect anomalies from image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        image_np = np.array(image)
        
        result = real_detector.analyze_frame(image_np)
        
        # Save to DB
        with SessionLocal() as session:
            for anomaly in result.get('anomalies', []):
                session.execute(text("""
                    INSERT INTO anomaly_events (track_id, anomaly_type, confidence, description)
                    VALUES (:track_id, :type, :confidence, :description)
                """), {
                    'track_id': anomaly.get('track_id', 1),
                    'type': anomaly.get('type', 'unknown')[:50],
                    'confidence': anomaly.get('confidence', 0.5),
                    'description': anomaly.get('description', 'No description')[:200]
                })
            session.commit()
        
        return DetectResponse(
            status="success",
            detection_type="REAL_YOLO",
            people_detected=result.get('people_detected', 0),
            anomalies_detected=result.get('anomalies_detected', 0),
            anomalies=result.get('anomalies', []),
            people_data=result.get('people_data', []),
            ml_models=HAS_ML_MODELS,
            calibration_status=result.get('calibration_status', {}),
            message=f"Analysis: {result.get('people_detected', 0)} people, {result.get('anomalies_detected', 0)} anomalies"
        )
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/anomalies", response_model=AnomaliesResponse)
async def get_anomalies(limit: int = 10):
    """Get list of anomalies"""
    try:
        with SessionLocal() as session:
            rows = session.execute(text("""
                SELECT * FROM anomaly_events ORDER BY timestamp DESC LIMIT :limit
            """), {'limit': limit}).fetchall()
        
        anomalies = [
            AnomalyEvent(
                id=row[0], track_id=row[1], anomaly_type=row[2],
                confidence=row[3], description=row[4], timestamp=row[5]
            ) for row in rows
        ]
        return AnomaliesResponse(total=len(anomalies), anomalies=anomalies)
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=StatusResponse)
async def system_status():
    """System status"""
    return {
        "system": "running",
        "ml_models": HAS_ML_MODELS,
        "calibration": real_detector.get_calibration_status(),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host=settings.api.host, port=settings.api.port)