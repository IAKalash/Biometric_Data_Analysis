import numpy as np
import time
from ultralytics import YOLO
from core_logic import GeometryEngine, AnomalyAuditor, PersonState, _ensure_python_types 
from config_opt import settings
from typing import Dict, Any
from loguru import logger

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
            
        results = self.model.track(
            frame, 
            persist=True, 
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
                    "box": box.xyxy[0].cpu().numpy(), # NumPy array для GeometryEngine
                    "kpts": kps # Keypoints object
                }
                
                features = self.auditor.compute_feature_vector(kps_data, state, self.geometry)
                state.add_height(features.get('height_cm', 0.0))
                
                if not self.geometry.is_calibrated:
                    if features.get('bbox_height_px', 0) > 0:
                        self.geometry.calibrate_simple(features['bbox_height_px'], settings.REFERENCE_HEIGHT_CM)

                alerts = self.auditor.audit(features)
                
                person_meta = {
                    "id": tid,
                    "box": [float(val) for val in box.xyxy[0].tolist()], 
                    "features": features # Уже Python-типы
                }
                
                if alerts:
                    for alert in alerts:
                        alert['conf'] = float(alert.get('conf', 0.0)) 
                        alert['track_id'] = tid
                        current_anomalies.append(alert)
                    person_meta['status'] = 'ANOMALY'
                else:
                    person_meta['status'] = 'OK'
                    
                people_meta.append(person_meta)
        
        final_response = {
            "processed_at": float(time.time()),
            "people_count": len(people_meta),
            "anomalies": current_anomalies,
            "meta": people_meta,
            "calibrated": self.geometry.is_calibrated
        }
        
        return _ensure_python_types(final_response)

tracker_service = TrackingService()