import numpy as np
import time
from ultralytics import YOLO
from core_logic import GeometryEngine, AnomalyAuditor, PersonState, _ensure_python_types 
from config_opt import settings
from typing import Dict, Any
from loguru import logger
import cv2 

# --- КОНФИГУРАЦИЯ СКЕЛЕТА (для рисования на сервере) ---
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), 
    (1, 2), (0, 1), (0, 2), (0, 3), (0, 4), (3, 5), (4, 6)
]
# --------------------------------------------------------

class TrackingService:
    def __init__(self):
        try:
            self.model = YOLO(settings.YOLO_MODEL)
            logger.info(f"YOLOv8-pose model '{settings.YOLO_MODEL}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
        
        self.geometry = GeometryEngine()
        self.auditor = AnomalyAuditor()
        
        # Хранилище состояний: {track_id: PersonState}
        self.states: Dict[int, PersonState] = {}
        self.gc_interval = 5.0 
        self.last_gc_time = time.time()
        self.frame_counter = 0 
        
    def _cleanup_states(self) -> None:
        cutoff_time = time.time() - 5.0
        keys_to_delete = [tid for tid, state in self.states.items() if state.last_seen < cutoff_time]
        for tid in keys_to_delete:
            del self.states[tid]
        if keys_to_delete:
            logger.debug(f"GC: Deleted {len(keys_to_delete)} old tracks.")


    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            return {"error": "Model not loaded"}
        
        self.frame_counter += 1
        
        # 1. Применяем гомографию (если включена)
        processed_frame = self.geometry.normalize_frame(frame)
        
        # 2. Логика пропуска кадров
        should_process = (self.frame_counter % (settings.SKIP_FRAMES + 1) == 0)
        
        if not should_process:
            if settings.SERVER_DEBUG_DISPLAY:
                # Отображаем просто кадр, что инференс пропущен
                vis_frame = processed_frame.copy()
                calib_status = "CALIBRATED" if self.geometry.is_calibrated else "CALIBRATING..."
                calib_color = (0, 255, 0) if self.geometry.is_calibrated else (0, 165, 255)
                
                cv2.putText(vis_frame, f"STATUS: SKIPPED", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(vis_frame, calib_status, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, calib_color, 2)
                
                cv2.imshow('Server Debug View', vis_frame)
                cv2.waitKey(1)
            
            return {"processed_at": float(time.time()), 
                    "people_count": len(self.states), 
                    "anomalies": [], 
                    "meta": [], 
                    "calibrated": self.geometry.is_calibrated,
                    "homography_enabled": settings.HOMOGRAPHY_ENABLED,
                    "skipped": True 
                    }
        
        # Кадр обрабатывается (Инференс)
            
        # 3. Трекинг на обработанном кадре
        results = self.model.track(
            processed_frame, 
            persist=True, 
            imgsz=settings.INFERENCE_SIZE, 
            conf=settings.CONF_THRESHOLD, 
            iou=0.5, 
            classes=0, 
            verbose=False
        )
        
        current_anomalies = []
        people_meta = []
        now = time.time()
        
        if now - self.last_gc_time > self.gc_interval:
            self._cleanup_states()
            self.last_gc_time = now

        # Инициализация кадра для отображения (если отладка включена)
        vis_frame = processed_frame.copy()
        if settings.SERVER_DEBUG_DISPLAY:
            calib_status = "CALIBRATED" if self.geometry.is_calibrated else "CALIBRATING..."
            calib_color = (0, 255, 0) if self.geometry.is_calibrated else (0, 165, 255)
            cv2.putText(vis_frame, calib_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, calib_color, 2)

        for result in results:
            if not result.boxes or not result.keypoints:
                continue
                
            for box, kps in zip(result.boxes, result.keypoints):
                if box.id is None or int(box.cls) != 0:
                    continue
                
                tid = int(box.id.item()) 

                if tid not in self.states:
                    self.states[tid] = PersonState(tid)
                    
                state = self.states[tid]
                state.last_seen = now
                
                kps_data = {
                    "box": box.xyxy[0].cpu().numpy(),
                    "kpts": kps
                }
                
                # 4. Расчет признаков 
                features = self.auditor.compute_feature_vector(kps_data, state, self.geometry)
                state.add_height(features.get('height_cm', 0.0))
                
                # Авто-калибровка
                if not self.geometry.is_calibrated:
                    if features.get('bbox_height_px', 0) > 0:
                        self.geometry.calibrate_simple(features['bbox_height_px'], settings.REFERENCE_HEIGHT_CM)

                # 5. Аудит аномалий
                alerts = self.auditor.audit(features)
                
                kpts_list = kps.xy[0].cpu().numpy().tolist() if hasattr(kps, 'xy') else []

                person_meta = {
                    "id": tid,
                    "box": [float(val) for val in box.xyxy[0].tolist()], 
                    "keypoints": kpts_list, 
                    "features": features 
                }
                
                color = (0, 255, 0)
                if alerts:
                    color = (0, 0, 255)
                    for alert in alerts:
                        alert['conf'] = float(alert.get('conf', 0.0)) 
                        alert['track_id'] = tid
                        current_anomalies.append(alert)
                    person_meta['status'] = 'ANOMALY'
                else:
                    person_meta['status'] = 'OK'
                    
                people_meta.append(person_meta)

                # --- Рисование на отладочном кадре сервера ---
                if settings.SERVER_DEBUG_DISPLAY:
                    # Координаты уже в пространстве vis_frame
                    x1, y1, x2, y2 = [int(i.item()) for i in box.xyxy[0]]
                    keypoints_int = kps.xy[0].cpu().numpy().astype(int)[:, :2]

                    # 1. Box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 2. Skeleton
                    for i, j in SKELETON_CONNECTIONS:
                        p1 = keypoints_int[i]
                        p2 = keypoints_int[j]
                        if p1[0] > 0 and p2[0] > 0:
                            cv2.line(vis_frame, tuple(p1), tuple(p2), color, 2)

                    # 3. Points
                    for x, y in keypoints_int:
                        if x > 0 and y > 0:
                            cv2.circle(vis_frame, (x, y), 4, (255, 0, 0), -1)

                    # 4. Text
                    status_text = f"ID {tid} | Status: {person_meta['status']}"
                    info_text = f"H={features.get('height_cm', 0):.1f}cm, Ratio={features.get('ratio_sh_h', 0):.2f}"

                    cv2.putText(vis_frame, status_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(vis_frame, info_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- Финальное отображение (после обработки всех людей) ---
        if settings.SERVER_DEBUG_DISPLAY:
            cv2.imshow('Server Debug View', vis_frame)
            cv2.waitKey(1) # Небольшая задержка для обновления окна
        
        final_response = {
            "processed_at": float(time.time()),
            "people_count": len(people_meta),
            "anomalies": current_anomalies,
            "meta": people_meta,
            "calibrated": self.geometry.is_calibrated,
            "homography_enabled": settings.HOMOGRAPHY_ENABLED,
            "skipped": False
        }
        
        return _ensure_python_types(final_response)

tracker_service = TrackingService()