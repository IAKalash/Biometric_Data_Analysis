# config.py
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

class DBSettings(BaseModel):
    url: str = Field(default="sqlite:///./anomalies.db", env="DB_URL")
    echo: bool = Field(default=False, env="DB_ECHO")

class ModelSettings(BaseModel):
    yolo_model: str = Field(default="yolov8n-pose.pt", env="YOLO_MODEL")
    confidence_threshold: float = Field(default=0.6, env="CONF_THRESHOLD")
    iou_threshold: float = Field(default=0.5, env="IOU_THRESHOLD")

    @validator('confidence_threshold', 'iou_threshold')
    def validate_thresholds(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v

class BiometricSettings(BaseModel):
    reference_profiles: Dict[str, Dict[str, List[float]]] = {
        "male": {"height_range": [165, 185], "shoulder_width_range": [40, 55]},
        "female": {"height_range": [155, 175], "shoulder_width_range": [35, 45]}
    }

class APISettings(BaseModel):
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")

class MLSettings(BaseModel):
    n_estimators: int = Field(default=100, env="ML_N_ESTIMATORS")
    min_samples: int = Field(default=10, env="ML_MIN_SAMPLES")

class Settings(BaseModel):
    db: DBSettings = DBSettings()
    models: ModelSettings = ModelSettings()
    biometric: BiometricSettings = BiometricSettings()
    api: APISettings = APISettings()
    ml: MLSettings = MLSettings()

settings = Settings()