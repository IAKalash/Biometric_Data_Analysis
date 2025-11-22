# real_detection.py
import cv2
import numpy as np
from ultralytics import YOLO
import math
import statistics
import time
from typing import Dict, List, Any, Optional
from loguru import logger
from config import settings

# === ML модели ===
try:
    from ml_models import MLModels
    ml_system = MLModels()
    HAS_ML_MODELS = True
    logger.info("ML models loaded successfully")
except Exception as e:
    HAS_ML_MODELS = False
    logger.warning(f"ML models not available: {e}")

# === Глобальная загрузка YOLO (один раз при старте) ===
pose_model = YOLO(settings.models.yolo_model)
logger.info(f"YOLO model '{settings.models.yolo_model}' loaded globally")

class RealAnomalyDetector:
    def __init__(self) -> None:
        self.pixel_to_cm: float = 0.7
        self.is_calibrated: bool = False
        self.height_samples: List[float] = []
        self.calibration_threshold: int = 8          # быстро калибруется за 5–10 сек
        self.last_calibration_time: float = 0.0

    def force_calibration(self, assumed_height: float = 170.0) -> None:
        """Принудительная калибровка по медиане собранных замеров"""
        if len(self.height_samples) < 3:
            return
        median_px = statistics.median(self.height_samples)
        if median_px <= 50:
            return
        self.pixel_to_cm = assumed_height / median_px
        self.is_calibrated = True
        logger.info(f"КАЛИБРОВКА ЗАВЕРШЕНА! 1px = {self.pixel_to_cm:.4f} см "
                    f"(на основе {len(self.height_samples)} замеров, предполагаемый рост: {assumed_height} см)")
        self.height_samples.clear()

    def auto_calibrate(self, person_data: Dict[str, Any]) -> None:
        """Автоматическая калибровка с учётом пола"""
        if self.is_calibrated:
            return

        height_px = person_data.get('biometrics', {}).get('bbox_height_px', 0.0)
        if not (150 < height_px < 700):
            return

        self.height_samples.append(height_px)
        if len(self.height_samples) > 30:
            self.height_samples = self.height_samples[-20:]

        if len(self.height_samples) >= self.calibration_threshold:
            if time.time() - self.last_calibration_time > 3.0:
                self.last_calibration_time = time.time()
                gender = person_data.get('gender', {}).get('gender', 'unknown')
                assumed = 175 if gender == 'male' else 162 if gender == 'female' else 170
                self.force_calibration(assumed)

    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        try:
            results = pose_model(frame, conf=settings.models.confidence_threshold)

            anomalies: List[Dict] = []
            people_data: List[Dict] = []

            for result in results:
                if result.boxes is None or result.keypoints is None:
                    continue

                for box, keypoints in zip(result.boxes, result.keypoints):
                    if int(box.cls) != 0:
                        continue

                    person = self._extract_person_data(box, keypoints, frame)
                    people_data.append(person)

                    # Rule-based аномалии
                    person_anomalies = self._check_anomalies(person)
                    anomalies.extend(person_anomalies)

                    # ML анализ
                    if HAS_ML_MODELS:
                        ml_result = ml_system.process_person(person['biometrics'])
                        person['gender'] = ml_result['gender']
                        anomalies.extend(ml_result['ml_anomalies'])
                    else:
                        person['gender'] = {'gender': 'unknown', 'confidence': 0.0, 'method': 'no_ml'}

                    # Автокалибровка (после ML, чтобы был пол)
                    if not self.is_calibrated:
                        self.auto_calibrate(person)

            return {
                "people_detected": len(people_data),
                "anomalies_detected": len(anomalies),
                "anomalies": self._serialize_anomalies(anomalies),
                "people_data": self._serialize_people_data(people_data),
                "calibration_status": self.get_calibration_status()
            }

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "error": str(e),
                "people_detected": 0,
                "anomalies_detected": 0,
                "anomalies": [],
                "people_data": [],
                "calibration_status": self.get_calibration_status()
            }

    def _extract_person_data(self, box: Any, keypoints: Any, frame: np.ndarray) -> Dict[str, Any]:
        bbox = box.xyxy[0].cpu().numpy().tolist()
        confidence = float(box.conf.cpu().numpy()[0])
        track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None

        kps = keypoints.xy[0].cpu().numpy() if keypoints.xy is not None else []
        kp_dict = {
            'nose': self._kp(kps, 0),
            'left_shoulder': self._kp(kps, 5),
            'right_shoulder': self._kp(kps, 6),
            'left_hip': self._kp(kps, 11),
            'right_hip': self._kp(kps, 12),
        }

        biometrics = self._calculate_biometrics(kp_dict, bbox)

        return {
            'bbox': bbox,
            'confidence': confidence,
            'track_id': track_id,
            'keypoints': kp_dict,
            'biometrics': biometrics
        }

    def _kp(self, kps: np.ndarray, idx: int) -> Optional[List[float]]:
        return [float(kps[idx][0]), float(kps[idx][1])] if len(kps) > idx and kps[idx][0] > 0 else None

    def _calculate_biometrics(self, kp: Dict, bbox: List[float]) -> Dict[str, float]:
        x1, y1, x2, y2 = bbox
        height_px = y2 - y1
        biometrics = {}

        if height_px > 50:
            biometrics['estimated_height_cm'] = height_px * self.pixel_to_cm
            biometrics['bbox_height_px'] = height_px

        if kp['left_shoulder'] and kp['right_shoulder']:
            w = math.dist(kp['left_shoulder'], kp['right_shoulder'])
            biometrics['shoulder_width_cm'] = w * self.pixel_to_cm

        if kp['left_hip'] and kp['right_hip']:
            w = math.dist(kp['left_hip'], kp['right_hip'])
            biometrics['hip_width_cm'] = w * self.pixel_to_cm

        return biometrics

    def _check_anomalies(self, person: Dict[str, Any]) -> List[Dict]:
        anomalies = []
        b = person['biometrics']

        h = b.get('estimated_height_cm', 0)
        if h > 0:
            if not self.is_calibrated:
                if h < 100 or h > 220:
                    anomalies.append({
                        'type': 'abnormal_height',
                        'confidence': 0.6,
                        'description': f'Possible abnormal height: {h:.1f} cm (calibrating)',
                        'severity': 'low'
                    })
            else:
                if h < 145:
                    anomalies.append({
                        'type': 'abnormal_height',
                        'confidence': min(0.95, (145 - h) / 20),
                        'description': f'Low height: {h:.1f} cm',
                        'severity': 'high'
                    })
                elif h > 195:
                    anomalies.append({
                        'type': 'abnormal_height',
                        'confidence': min(0.95, (h - 195) / 20),
                        'description': f'High height: {h:.1f} cm',
                        'severity': 'high'
                    })

        s = b.get('shoulder_width_cm', 0)
        if s > 65:
            anomalies.append({
                'type': 'abnormal_shoulders',
                'confidence': min(0.9, (s - 65) / 20),
                'description': f'Wide shoulders: {s:.1f} cm',
                'severity': 'medium'
            })

        hip = b.get('hip_width_cm', 0)
        if hip > 0 and (hip < 25 or hip > 70):
            anomalies.append({
                'type': 'abnormal_hips',
                'confidence': 0.7,
                'description': f'Abnormal hip width: {hip:.1f} cm',
                'severity': 'medium'
            })

        return anomalies

    def _serialize_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        return [{k: v for k, v in a.items() if k in {'type', 'confidence', 'description', 'severity'}} for a in anomalies]

    def _serialize_people_data(self, people: List[Dict]) -> List[Dict]:
        return [{
            'bbox': p['bbox'],
            'confidence': p['confidence'],
            'track_id': p['track_id'],
            'biometrics': {k: round(v, 2) for k, v in p['biometrics'].items()},
            'gender': p.get('gender', {'gender': 'unknown', 'confidence': 0.0})
        } for p in people]

    def get_calibration_status(self) -> Dict[str, Any]:
        return {
            'calibrated': self.is_calibrated,
            'pixel_to_cm': round(self.pixel_to_cm, 4),
            'samples_collected': len(self.height_samples),
            'status': 'calibrated' if self.is_calibrated else 'collecting_samples'
        }

# Глобальный экземпляр
real_detector = RealAnomalyDetector()