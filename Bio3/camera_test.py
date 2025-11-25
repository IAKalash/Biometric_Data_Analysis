# camera_test.py - НОВАЯ ВЕРСИЯ
import cv2
import requests
import numpy as np
import time
import os
import sys
from datetime import datetime
from loguru import logger
from typing import Dict, Any, List, Tuple

# --- Конфигурация ---
API_URL = "http://localhost:8000/analyze_frame"
STATUS_URL = "http://localhost:8000/status"
ANOMALY_DIR = "anomalies"
os.makedirs(ANOMALY_DIR, exist_ok=True)
logger.info(f"Аномалии будут сохраняться в папку: {ANOMALY_DIR}/")

# --- Вспомогательные функции ---
def get_api_status() -> Dict[str, Any]:
    """Получает статус системы, включая калибровку, через /status."""
    try:
        r = requests.get(STATUS_URL, timeout=1)
        if r.status_code == 200:
            status_data = r.json()
            # В service.py мы возвращаем 'calibrated'
            return {
                'calibration': status_data.get('calibrated', False), 
                # Извлекаем px_to_cm_ratio из логики GeometryEngine, 
                # но пока API возвращает только 'calibrated', 
                # используем заглушку, пока не обновим /status в main.py
                'px_to_cm_ratio': 0.0 
            }
    except requests.exceptions.RequestException:
        return {'calibration': False, 'px_to_cm_ratio': 0.0}
    return {'calibration': False, 'px_to_cm_ratio': 0.0}

def process_frame(frame: np.ndarray):
    """Отправляет кадр в API и получает результат."""
    try:
        # Ресайз до 640x480 для стандартизации и уменьшения нагрузки
        frame_resized = cv2.resize(frame, (640, 480))
        _, img_encoded = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
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

def draw_info(frame: np.ndarray, result: Dict, status: Dict) -> Tuple[np.ndarray, List]:
    """Рисует на кадре BBox, аномалии и статус калибровки."""
    
    # 1. Статус калибровки
    calibrated = status.get('calibration', False)
    status_text = "STATUS: CALIBRATED" if calibrated else "STATUS: CALIBRATING..."
    color = (0, 255, 0) if calibrated else (0, 165, 255) # Зеленый / Оранжевый
    
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # 2. Обводка людей
    people_data = result.get('meta', [])
    anomalies_list = result.get('anomalies', [])
    
    for person in people_data:
        bbox = person.get('box', [])
        person_id = person.get('id', -1)
        
        if len(bbox) == 4:
            # Масштабируем BBox обратно, т.к. frame не был ресайзнут
            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 480
            
            x1, y1, x2, y2 = [
                int(bbox[0] * scale_x), int(bbox[1] * scale_y), 
                int(bbox[2] * scale_x), int(bbox[3] * scale_y)
            ]
            
            h_cm = person.get('features', {}).get('height_cm', 0.0)
            
            is_anomaly = person.get('status') == 'ANOMALY'
            box_color = (0, 0, 255) if is_anomaly else (255, 0, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            
            label = f"ID:{person_id} H:{h_cm:.0f}cm" if h_cm > 0 else f"ID:{person_id} (Calibrating)"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # 3. Аномалии (отображаются в верхнем правом углу)
    for i, anomaly in enumerate(anomalies_list):
        # Используем .get() на случай, если поле отсутствует
        text = f"ANOMALY: {anomaly.get('type', 'Unknown')} ({anomaly.get('conf', 0.0):.2f})"
        cv2.putText(frame, text, (frame.shape[1] - 400, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    return frame, anomalies_list

# --- Основной цикл ---
def run_camera_test():
    # cv2.CAP_ANY - пытается использовать лучший доступный бэкенд
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2) 
    if not cap.isOpened():
        logger.error("Не могу открыть камеру. Проверьте подключение и права. Попробуйте cv2.VideoCapture(0) без второго аргумента.")
        return

    logger.info(f"API URL: {API_URL}. Убедитесь, что main.py запущен.")
    logger.info("Нажмите 'q' для выхода. Подождите 10-20 кадров, чтобы завершить калибровку.")
    
    last_anomaly_save_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Не удалось получить кадр с камеры.")
            break
        
        # 1. Обработка кадра
        api_result = process_frame(frame.copy()) 
        status = get_api_status() # Получаем статус из API

        # 2. Отображение
        frame_to_show = frame
        anomalies_list = []
        if api_result:
            frame_to_show, anomalies_list = draw_info(frame, api_result, status)
            
            # 3. Сохранение аномалий
            if anomalies_list and time.time() - last_anomaly_save_time > 1.0:
                logger.warning(f"Anomalies detected: {len(anomalies_list)}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(ANOMALY_DIR, f"anomaly_{timestamp}.jpg")
                cv2.imwrite(filename, frame) # Сохраняем исходный кадр
                last_anomaly_save_time = time.time()
                
        else:
            # Если API не отвечает, все равно отображаем статус (хотя он будет 'CALIBRATING')
            draw_info(frame, {'meta': [], 'anomalies': []}, status)
            
        cv2.imshow('BioAnomaly Detection', frame_to_show)

        # 4. Выход по 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Camera test terminated.")

if __name__ == "__main__":
    run_camera_test()