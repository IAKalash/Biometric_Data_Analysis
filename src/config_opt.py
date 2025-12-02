from pydantic_settings import BaseSettings
from typing import List, Tuple

class Settings(BaseSettings):
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # YOLO
    YOLO_MODEL: str = "models/yolov8n-pose.pt"
    INFERENCE_SIZE: int = 2880       # Размер для модели YOLO
    CONF_THRESHOLD: float = 0.4       # Порог уверенности
    SKIP_FRAMES: int = 10            # Пропускать N кадров для ускорения
    
    HOMOGRAPHY_ENABLED: bool = True
    HOMOGRAPHY_MATRIX_PATH: str = "models/H_cam_01.json"
    TARGET_SIZE: Tuple[int, int] = (3000, 1500)

    SERVER_DEBUG_DISPLAY: bool = True
    
    # Биометрические эталоны (М/Ж объединены в диапазоны для векторизации)
    # [min, max]
    REF_HEIGHT: List[float] = [145, 195] 
    REF_SHOULDER_RATIO: List[float] = [0.22, 0.30]
    REFERENCE_HEIGHT_CM: float = 175.0 # Средний рост для авто-калибровки
    
    # Пороги аномалий
    ANOMALY_SCORE_THRESHOLD: float = 0.75  # Комбинированный порог
    SPEED_ANOMALY_THRESHOLD: float = 150.0 # см/с

    class Config:
        env_file = ".env"

settings = Settings()