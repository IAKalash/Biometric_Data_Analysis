import numpy as np
from collections import deque
import statistics 
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from loguru import logger
from config_opt import settings
import math 

# Утилита для конвертации NumPy-типов
def _ensure_python_types(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: _ensure_python_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_ensure_python_types(v) for v in data]
    if isinstance(data, (np.float32, np.float64, np.number)):
        return float(data)
    if isinstance(data, (np.int32, np.int64)):
        return int(data)
    if hasattr(data, 'tolist') and not isinstance(data, str):
        try:
            return data.tolist()
        except Exception:
            pass
    return data

@dataclass
class PersonState:
    track_id: int
    first_seen: float
    last_seen: float
    pos_history: deque = None   
    height_history: deque = None 
    
    def __init__(self, track_id):
        self.track_id = track_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.pos_history = deque(maxlen=15)
        self.height_history = deque(maxlen=30)
        
    def add_position(self, x: float, y: float, timestamp: float):
        self.pos_history.append((float(x), float(y), float(timestamp))) 
        self.last_seen = float(timestamp)

    def add_height(self, h_cm: float):
        self.height_history.append(float(h_cm)) 

# Модуль Нормализации (Geometry Engine)
class GeometryEngine:
    def __init__(self):
        self.homography_matrix = None
        self.px_to_cm_ratio = 0.0
        self.is_calibrated = False
        self.height_samples: deque = deque(maxlen=30) 
        self.last_calib_time = time.time()
        
    def calibrate_simple(self, px_height: float, ref_height_cm: float = 175.0):
        """Простая линейная калибровка"""
        if px_height <= 10: return

        self.height_samples.append(float(px_height)) 
        
        if len(self.height_samples) % 10 == 0:
            logger.debug(f"Calib samples: {len(self.height_samples)}. Last px: {px_height:.1f}")

        if len(self.height_samples) >= 15 and (time.time() - self.last_calib_time > 1.0):
            median_px = statistics.median(self.height_samples)
            
            # Защита от слишком маленьких объектов (менее 10% кадра)
            if median_px > 100: 
                self.px_to_cm_ratio = float(ref_height_cm) / float(median_px) 
                self.is_calibrated = True
                self.last_calib_time = time.time()
                self.height_samples.clear()
                logger.success(f"CALIBRATION DONE! 1px = {self.px_to_cm_ratio:.4f} cm. (Median Height: {median_px:.1f}px)")
            else:
                # Если медиана слишком маленькая - человек далеко или ошибка детекции. Сбрасываем.
                if len(self.height_samples) >= 29:
                    self.height_samples.clear()
                    logger.warning(f"Calibration reset: object too small ({median_px:.1f}px). Step closer.")

    def get_features(self, kps: np.ndarray, bbox: np.ndarray) -> Tuple[Dict, Optional[Tuple[float, float]]]:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = y2 
        height_px = y2 - y1
        
        shoulder_l = self._kp(kps, 5)
        shoulder_r = self._kp(kps, 6)
        
        shoulder_width_px = math.dist(shoulder_l, shoulder_r) if shoulder_l and shoulder_r else 0.0
        
        features = {
            "bbox_height_px": float(height_px),
            "shoulder_width_px": float(shoulder_width_px),
            "height_cm": 0.0,
            "shoulder_width_cm": 0.0,
        }
        
        metric_center = None
        
        if self.is_calibrated and self.px_to_cm_ratio > 0:
            ratio = self.px_to_cm_ratio
            features['height_cm'] = float(height_px * ratio) 
            features['shoulder_width_cm'] = float(shoulder_width_px * ratio) 
            metric_center = (float(center_x * ratio), float(center_y * ratio))
        
        return features, metric_center

    def _kp(self, kps: np.ndarray, idx: int) -> Optional[Tuple[float, float]]:
        if kps.size > idx and kps[idx][0] > 0 and kps[idx][1] > 0:
            return (float(kps[idx][0]), float(kps[idx][1]))
        return None

class AnomalyAuditor:
    def __init__(self):
        pass

    def compute_feature_vector(self, kps: Dict, state: PersonState, geometry: GeometryEngine) -> Dict[str, Any]:
        bbox_np = kps['box'] 
        kps_np = kps['kpts'].xy[0].cpu().numpy() if hasattr(kps.get('kpts'), 'xy') else np.array([])
        
        features, metric_center = geometry.get_features(kps_np, bbox_np)
        
        h = features.get('height_cm', 0.0)
        s = features.get('shoulder_width_cm', 0.0)

        # Расчет скорости
        speed_cm_s = 0.0
        now = time.time()
        
        if metric_center and len(state.pos_history) > 0 and geometry.px_to_cm_ratio > 0:
            last_x, last_y, last_t = state.pos_history[-1]
            dt = now - last_t
            if dt > 0:
                dist_cm = math.hypot(metric_center[0] - last_x, metric_center[1] - last_y)
                speed_cm_s = dist_cm / dt
        
        # Обновляем историю позиции
        if metric_center:
            state.add_position(metric_center[0], metric_center[1], now)

        ratio_sh_h = float(s / h) if h > 0 else 0.0 
        
        feature_vector = {
            "height_cm": float(h), 
            "shoulder_width_cm": float(s),
            "ratio_sh_h": ratio_sh_h,
            "speed_cm_s": float(speed_cm_s),
            "bbox_height_px": features['bbox_height_px']
        }
        
        return _ensure_python_types(feature_vector)

    def audit(self, features: Dict[str, Any]) -> List[Dict]:
        anomalies = []
        h = features['height_cm']
        s = features['shoulder_width_cm']
        r = features['ratio_sh_h']
        v = features['speed_cm_s']
        
        if h < 50: return [] 

        if r > 0.35: 
            anomalies.append({
                "type": "MASKING_RISK",
                "conf": 0.85, 
                "desc": f"Disproportionate shoulders ({s:.1f}cm) for height ({h:.1f}cm)"
            })
            
        if v > settings.SPEED_ANOMALY_THRESHOLD:
            conf_val = float(min(v / 400.0, 0.99))
            anomalies.append({
                "type": "HIGH_SPEED",
                "conf": conf_val, 
                "desc": f"Unnatural speed: {v:.0f} cm/s"
            })
            
        return _ensure_python_types(anomalies)