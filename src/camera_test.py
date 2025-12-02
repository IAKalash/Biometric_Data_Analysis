import cv2
import requests
import numpy as np
import time
import os
from datetime import datetime
from loguru import logger
from typing import Dict, Any, List, Tuple

SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), 
    (1, 2), (0, 1), (0, 2), (0, 3), (0, 4), (3, 5), (4, 6)
]
LINE_COLOR = (255, 0, 255)
POINT_COLOR = (0, 255, 255)

API_URL = "http://localhost:8000/analyze_frame"
STATUS_URL = "http://localhost:8000/status"
ANOMALY_DIR = "anomalies"
os.makedirs(ANOMALY_DIR, exist_ok=True)
logger.info(f"Аномалии будут сохраняться в папку: {ANOMALY_DIR}/")

def get_api_status() -> Dict[str, Any]:
    """Получает статус калибровки и соотношение px/cm с сервера."""
    try:
        r = requests.get(STATUS_URL, timeout=1)
        if r.status_code == 200:
            status_data = r.json()
            return {
                'calibration': status_data.get('calibrated', False), 
                'px_to_cm_ratio': status_data.get('px_to_cm', 0.0)
            }
    except requests.exceptions.RequestException:
        pass
    return {'calibration': False, 'px_to_cm_ratio': 0.0}

def process_frame(frame: np.ndarray):
    """ИСПРАВЛЕНО: Отправляет полноразмерный кадр на сервер."""
    try:
        _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        r = requests.post(API_URL, files=files, timeout=5)
        
        if r.status_code == 200:
            return r.json()
        
        logger.warning(f"API returned status {r.status_code}: {r.text[:100]}")
    except requests.exceptions.RequestException as e:
        logger.debug(f"API communication error: {e}")
    except Exception as e:
        logger.error(f"Error during frame processing: {e}")
    return None

def draw_info(frame: np.ndarray, meta: List[Dict], status: Dict, mode_text: str) -> np.ndarray:
    """Рисует данные о людях (bbox, скелет) и статус на кадре."""
    
    vis_frame = frame.copy()

    calibrated = status.get('calibration', False)
    ratio = status.get('px_to_cm_ratio', 0.0)
    status_text = f"Calibration: {'CALIBRATED' if calibrated else 'CALIBRATING...'} (1px={ratio:.3f}cm)"
    color = (0, 255, 0) if calibrated else (0, 165, 255)
    
    cv2.putText(vis_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    mode_color = (0, 255, 0) if mode_text == 'PROCESSED' else (0, 165, 255)
    cv2.putText(vis_frame, f"Mode: {mode_text}", (vis_frame.shape[1] - 300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    
    for person in meta:
        bbox = person.get('box', [])
        person_id = person.get('id', -1)
        keypoints = person.get('keypoints', [])
        features = person.get('features', {})
        
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            
            h_cm = features.get('height_cm', 0.0)
            
            is_anomaly = person.get('status') == 'ANOMALY'
            box_color = (0, 0, 255) if is_anomaly else (255, 0, 0) # BGR
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), box_color, 2)
            
            if keypoints:
                keypoints_int = np.array(keypoints).astype(int)
                
                for start_idx, end_idx in SKELETON_CONNECTIONS:
                    if start_idx < len(keypoints_int) and end_idx < len(keypoints_int):
                        p1 = keypoints_int[start_idx]
                        p2 = keypoints_int[end_idx]
                        if p1[0] > 0 and p2[0] > 0:
                            cv2.line(vis_frame, (int(p1[0]), int(p1[1])), 
                                    (int(p2[0]), int(p2[1])), LINE_COLOR, 2)

                for x, y in keypoints_int:
                    if x > 0 and y > 0:
                        cv2.circle(vis_frame, (int(x), int(y)), 4, POINT_COLOR, -1)
            
            label = f"ID:{person_id} H:{h_cm:.0f}cm" if h_cm > 0 else f"ID:{person_id} (Calibrating)"
            cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
    return vis_frame

def run_camera_test():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
    if not cap.isOpened():
        logger.error("Не могу открыть камеру. Проверьте подключение и права. Попробуйте cv2.VideoCapture(0) без второго аргумента.")
        return

    logger.info(f"API URL: {API_URL}. Убедитесь, что main.py запущен.")
    logger.info("Нажмите 'q' для выхода. Кадры отображаются с интерполяцией между обработками.")
    
    last_anomaly_save_time = time.time()
    
    last_valid_meta: List[Dict] = []
    last_status: Dict = get_api_status() 
    last_anomalies: List[Dict] = [] 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Не удалось получить кадр с камеры.")
            break

        api_result = process_frame(frame.copy()) 
        current_status = get_api_status()
        last_status = current_status 

        meta_to_draw = []
        mode_text = "INTERPOLATING"
        anomalies_to_show = last_anomalies 

        if api_result:
            if not api_result.get('skipped', False):
                last_valid_meta = api_result.get('meta', [])
                last_anomalies = api_result.get('anomalies', [])
                meta_to_draw = last_valid_meta
                anomalies_to_show = last_anomalies
                mode_text = "PROCESSED"
            else:
                meta_to_draw = last_valid_meta
                
            frame_to_show = draw_info(frame, meta_to_draw, last_status, mode_text)
            
            for i, anomaly in enumerate(anomalies_to_show):
                text = f"ANOMALY: {anomaly.get('type', 'Unknown')} ({anomaly.get('conf', 0.0):.2f}) ID:{anomaly.get('track_id', -1)}"
                cv2.putText(frame_to_show, text, (frame_to_show.shape[1] - 450, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if last_anomalies and mode_text == 'PROCESSED' and time.time() - last_anomaly_save_time > 1.0:
                logger.warning(f"Anomalies detected: {len(last_anomalies)}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(ANOMALY_DIR, f"anomaly_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                last_anomaly_save_time = time.time()
                
        else:
            frame_to_show = draw_info(frame, last_valid_meta, last_status, "API_ERROR")
            cv2.putText(frame_to_show, "API ERROR", 
                        (frame_to_show.shape[1] - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow('BioAnomaly Detection', frame_to_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Camera test terminated.")

if __name__ == "__main__":
    run_camera_test()