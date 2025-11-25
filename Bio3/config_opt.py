import os
from pydantic_settings import BaseSettings
from pydantic import Field

from typing import Dict, List

class Settings(BaseSettings):
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # YOLO
    YOLO_MODEL: str = "yolov8n-pose.pt"
    CONF_THRESHOLD: float = 0.5
    
    # Нормализация и Гомография
    # 4 точки на полу в кадре (src) -> 4 точки в реальном мире (dst, метры)
    # Если None - используется простая линейная калибровка по росту
    USE_HOMOGRAPHY: bool = False 
    
    # Биометрические эталоны (М/Ж объединены в диапазоны для векторизации)
    # [min, max]
    REF_HEIGHT: List[float] = [145, 195] 
    REF_SHOULDER_RATIO: List[float] = [0.22, 0.30]
    REFERENCE_HEIGHT_CM: float = 175.0 # ширина плеч / рост
    
    # Пороги аномалий
    ANOMALY_SCORE_THRESHOLD: float = 0.75  # Комбинированный порог
    SPEED_ANOMALY_THRESHOLD: float = 150.0 # см/с

    class Config:
        env_file = ".env"

settings = Settings()