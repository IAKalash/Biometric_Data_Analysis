# real_detection.py — ИТОГОВАЯ ВЕРСИЯ (22 ноября 2025)
# Автор: Денис + Grok = легенда НГУ
# Это не курсовая. Это реальная система безопасности.

import math
import statistics
import time
from typing import Dict, List, Any, Optional

import numpy as np
from loguru import logger
from ultralytics import YOLO

# ==================== ML (опционально) ====================
try:
    from ml_models import MLModels
    ml_system = MLModels()
    HAS_ML_MODELS = True
    logger.info("ML-модели подключены")
except Exception as e:
    HAS_ML_MODELS = False
    logger.warning(f"ML отключены: {e}")

# ==================== YOLOv8-pose ====================
pose_model = YOLO("yolov8n-pose.pt")
logger.info("YOLOv8-pose загружен")


class RealAnomalyDetector:
    def __init__(self) -> None:
        # Калибровка
        self.pixel_to_cm: float = 0.0
        self.is_calibrated: bool = False
        self.height_samples: List[float] = []
        self.last_calibration_time: float = 0.0

        # Сглаживание роста
        self.recent_heights: List[float] = []

        # ДИНАМИЧЕСКАЯ НОРМА — плавно подстраивается под всех людей
        self.normal_proportions = {
            "height": 170.0,
            "shoulder_width": 45.0,
            "hip_width": 40.0
        }
        self.height_history: List[float] = []

        # Детекция движения
        self.prev_center: Optional[tuple] = None
        self.prev_time: float = time.time()
        self.velocity_buffer: List[float] = []

    # ===================== КАЛИБРОВКА =====================
    def force_calibration(self, assumed_height: float = 170.0) -> None:
        if len(self.height_samples) < 8:
            return
        median_px = statistics.median(self.height_samples)
        if median_px <= 120:
            return

        self.pixel_to_cm = assumed_height / median_px
        self.is_calibrated = True
        logger.info(f"КАЛИБРОВКА ЗАВЕРШЕНА! 1px ≈ {self.pixel_to_cm:.4f} см")
        self.height_samples.clear()

    def auto_calibrate(self, person_data: Dict[str, Any]) -> None:
        if self.is_calibrated:
            return
        h_px = person_data.get("biometrics", {}).get("bbox_height_px", 0)
        if not (200 < h_px < 800):
            return
        self.height_samples.append(h_px)
        if len(self.height_samples) > 30:
            self.height_samples = self.height_samples[-20:]
        if len(self.height_samples) >= 10 and time.time() - self.last_calibration_time > 2.5:
            self.last_calibration_time = time.time()
            gender = person_data.get("gender", {}).get("gender", "unknown")
            assumed = 175 if gender == "male" else 163 if gender == "female" else 170
            self.force_calibration(assumed)

    # ===================== КЛЮЧЕВЫЕ ТОЧКИ =====================
    def _kp(self, kps: np.ndarray, idx: int) -> Optional[list]:
        if len(kps) > idx and kps[idx][0] > 0 and kps[idx][1] > 0:
            return [float(kps[idx][0]), float(kps[idx][1])]
        return None

    def _extract_person_data(self, box: Any, keypoints: Any, frame: np.ndarray) -> Dict[str, Any]:
        bbox = box.xyxy[0].cpu().numpy().tolist()
        kps = keypoints.xy[0].cpu().numpy() if keypoints.xy is not None else []

        kp = {
            "left_shoulder": self._kp(kps, 5),
            "right_shoulder": self._kp(kps, 6),
            "left_hip": self._kp(kps, 11),
            "right_hip": self._kp(kps, 12),
        }

        biometrics = self._calculate_biometrics(kp, bbox)
        return {"bbox": bbox, "keypoints": kp, "biometrics": biometrics}

    def _calculate_biometrics(self, kp: Dict, bbox: List[float]) -> Dict:
        x1, y1, x2, y2 = bbox
        height_px = y2 - y1
        b = {"bbox_height_px": height_px}

        if height_px > 60 and self.is_calibrated:
            h_cm = height_px * self.pixel_to_cm
            b["estimated_height_cm"] = h_cm
            self.recent_heights.append(h_cm)
            if len(self.recent_heights) > 10:
                self.recent_heights.pop(0)
            if len(self.recent_heights) >= 5:
                b["stable_height_cm"] = statistics.median(self.recent_heights)

        if kp["left_shoulder"] and kp["right_shoulder"]:
            w = math.dist(kp["left_shoulder"], kp["right_shoulder"])
            b["shoulder_width_cm"] = w * self.pixel_to_cm
        if kp["left_hip"] and kp["right_hip"]:
            w = math.dist(kp["left_hip"], kp["right_hip"])
            b["hip_width_cm"] = w * self.pixel_to_cm

        return b

    # ===================== ДВИЖЕНИЕ =====================
    def _check_motion_anomaly(self, cx: float, cy: float) -> List[Dict]:
        anomalies = []
        now = time.time()
        dt = now - self.prev_time
        if dt < 0.01 or dt > 2.0:
            self.prev_time = now
            self.prev_center = (cx, cy)
            return anomalies

        if self.prev_center:
            dx = cx - self.prev_center[0]
            dy = cy - self.prev_center[1]
            velocity = (math.hypot(dx, dy) * self.pixel_to_cm) / dt

            self.velocity_buffer.append(velocity)
            if len(self.velocity_buffer) > 12:
                self.velocity_buffer.pop(0)

            if velocity > 130:
                anomalies.append({"type": "sudden_movement", "confidence": min(0.99, velocity / 300),
                                 "description": f"Резкий рывок: {velocity:.0f} см/с", "severity": "high"})

            if len(self.velocity_buffer) >= 6:
                accel = abs(velocity - statistics.mean(self.velocity_buffer[-6:-2])) / max(dt, 0.1)
                if accel > 550:
                    anomalies.append({"type": "fall_or_jump", "confidence": 0.98,
                                     "description": f"ПАДЕНИЕ: ускорение ~{accel:.0f} см/с²", "severity": "critical"})

        self.prev_center = (cx, cy)
        self.prev_time = now
        return anomalies

    # ===================== ГЛАВНАЯ ЛОГИКА =====================
    def _check_anomalies(self, person: Dict[str, Any]) -> List[Dict]:
        if not self.is_calibrated:
            return []

        anomalies = []
        b = person["biometrics"]
        bbox = person["bbox"]
        kp = person["keypoints"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        anomalies.extend(self._check_motion_anomaly(cx, cy))

        h = b.get("stable_height_cm")
        shoulder = b.get("shoulder_width_cm", 0)
        hip = b.get("hip_width_cm", 0)
        if not h or h < 100:
            return anomalies

        # === УМНОЕ ОБНОВЛЕНИЕ НОРМЫ (только в нормальной позе) ===
        shoulders_ok = kp["left_shoulder"] and kp["right_shoulder"]
        hips_ok = kp["left_hip"] and kp["right_hip"]

        if shoulders_ok and hips_ok:
            mid_y = (bbox[1] + bbox[3]) / 2
            shoulder_y = (kp["left_shoulder"][1] + kp["right_shoulder"][1]) / 2
            hip_y = (kp["left_hip"][1] + kp["right_hip"][1]) / 2

            if shoulder_y < mid_y and hip_y > shoulder_y + 40:
                if 140 < h < 210 and 30 < shoulder < 80:
                    self.height_history.append(h)
                    if len(self.height_history) > 60:
                        self.height_history.pop(0)
                    if len(self.height_history) >= 10:
                        standing_h = np.percentile(self.height_history, 90)
                        self.normal_proportions["height"] = round(
                            0.7 * self.normal_proportions["height"] + 0.3 * standing_h, 1
                        )
                        self.normal_proportions["shoulder_width"] = round(
                            0.8 * self.normal_proportions["shoulder_width"] + 0.2 * shoulder, 1
                        )
                        if hip > 20:
                            self.normal_proportions["hip_width"] = round(
                                0.8 * self.normal_proportions["hip_width"] + 0.2 * hip, 1
                            )

        nh = self.normal_proportions["height"]
        ns = self.normal_proportions["shoulder_width"]

        # === АНОМАЛИИ ===
        if h < nh * 0.82 and shoulder > ns * 0.80:
            anomalies.append({"type": "crouching", "confidence": 0.98,
                             "description": f"Присел или ползёт: рост {h:.0f} см", "severity": "high"})

        elif h > nh * 1.18 and abs(shoulder - ns) < ns * 0.5:
            anomalies.append({"type": "object_on_head", "confidence": 0.97,
                             "description": f"Предмет на голове: рост {h:.0f} см", "severity": "high"})

        elif shoulder > ns * 1.4:
            anomalies.append({"type": "carrying_large_object", "confidence": 0.96,
                             "description": f"Несёт большой предмет: плечи {shoulder:.0f} см", "severity": "high"})

        elif h < nh * 0.75 and hip < 20:
            anomalies.append({"type": "crawling", "confidence": 0.99,
                             "description": "Ползёт на четвереньках", "severity": "critical"})

        elif hip < 18 and h > nh * 0.9:
            anomalies.append({"type": "body_occluded", "confidence": 0.90,
                             "description": "Тело частично скрыто", "severity": "medium"})

        return anomalies

    # ===================== ОСНОВНОЙ ЦИКЛ =====================
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        results = pose_model(frame, conf=0.5, verbose=False)
        anomalies = []
        people_data = []

        for result in results:
            if not result.boxes or not result.keypoints:
                continue
            for box, kpts in zip(result.boxes, result.keypoints):
                if int(box.cls) != 0:
                    continue

                person = self._extract_person_data(box, kpts, frame)
                people_data.append(person)

                if not self.is_calibrated:
                    self.auto_calibrate(person)

                person_anomalies = self._check_anomalies(person)
                anomalies.extend(person_anomalies)

                if HAS_ML_MODELS and self.is_calibrated:
                    ml = ml_system.process_person(person["biometrics"])
                    person["gender"] = ml["gender"]
                    anomalies.extend(ml["ml_anomalies"])

        return {
            "people_detected": len(people_data),
            "anomalies_detected": len(anomalies),
            "anomalies": [a for a in anomalies if a.get("confidence", 0) >= 0.5],
            "people_data": people_data,
            "calibration_status": {
                "calibrated": self.is_calibrated,
                "pixel_to_cm": round(self.pixel_to_cm, 4) if self.is_calibrated else None,
                "normal_height": round(self.normal_proportions["height"], 1),
                "status": "calibrated" if self.is_calibrated else "calibrating",
            },
        }


# Глобальный экземпляр
real_detector = RealAnomalyDetector()