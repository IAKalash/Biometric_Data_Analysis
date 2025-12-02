import cv2
import requests
import numpy as np
import time
import os
import json
from datetime import datetime
from loguru import logger
from typing import Dict, Any, List, Tuple, Optional

# --- КОНФИГУРАЦИЯ СКЕЛЕТА (для рисования) ---
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), 
    (1, 2), (0, 1), (0, 2), (0, 3), (0, 4), (3, 5), (4, 6)
]

# ================= КОНФИГУРАЦИЯ КЛИЕНТА =================
VIDEO_FILENAME = "video.webm" 
DATA_DIR = "data"
VIDEO_PATH = os.path.join(DATA_DIR, VIDEO_FILENAME)

# Настройки API
API_URL = "http://localhost:8000/analyze_frame"
STATUS_URL = "http://localhost:8000/status"
ANOMALY_DIR = "anomalies"
os.makedirs(ANOMALY_DIR, exist_ok=True)

# Настройки Homography (для обратной проекции)
HOMOGRAPHY_MATRIX_PATH = "models/H_cam_01.json" 
HOMOGRAPHY_ENABLED = True

# Настройки Отображения
# Клиент отправляет полноразмерный кадр.
DISPLAY_MAX_WIDTH = 1280 # Максимальная ширина для отображения на экране
# ===============================================

logger.info(f"Аномалии будут сохраняться в папку: {ANOMALY_DIR}/")

# --- УТИЛИТЫ ГОМОГРАФИИ ---

def load_homography_matrix(path: str) -> Optional[np.ndarray]:
    """Загружает матрицу H из JSON."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            H_list = json.load(f)
        return np.array(H_list, dtype=np.float64)
    except Exception as e:
        logger.warning(f"Homography matrix not loaded/found at {path}: {e}")
        return None

def project_points(points: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    """
    Проецирует точки из нормированного пространства (сервера)
    обратно в исходное пространство (камеры) с помощью H_inv.
    """
    if H_inv is None or points.size == 0:
        return points.astype(int) 
        
    # Преобразование [x, y] в гомогенные координаты [x, y, 1]
    ones = np.ones((points.shape[0], 1))
    pts_homogeneous_warped = np.hstack([points, ones])
    
    # Применяем обратную матрицу: H_inv * P_warped
    pts_homogeneous_original = H_inv @ pts_homogeneous_warped.T
    
    # Нормализация (деление на третий компонент)
    pts_original = pts_homogeneous_original[:2] / pts_homogeneous_original[2]
    
    # Транспонирование обратно в формат (N, 2) и округление
    return pts_original.T.astype(int)

# --- Вспомогательные функции API ---

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

def process_frame(frame: np.ndarray) -> Optional[Dict[str, Any]]:
    """Кодирует и отправляет полноразмерный кадр на сервер для анализа."""
    try:
        # Кодирование полноразмерного кадра в JPEG для отправки
        _, img_encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        r = requests.post(API_URL, files=files, timeout=5)
        
        if r.status_code == 200:
            return r.json()
        else:
            logger.error(f"API Error: {r.status_code}, {r.text}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None

# --- ФУНКЦИЯ РИСОВАНИЯ (ОБНОВЛЕНА) ---

def draw_info(frame: np.ndarray, meta: List[Dict], status: Dict, H_inv: Optional[np.ndarray]) -> np.ndarray:
    """Рисует результаты на исходном кадре (с обратной проекцией)."""
    vis_frame = frame.copy()
    
    # 1. Информация о калибровке
    calib_status = "CALIBRATED" if status.get('calibration') else "CALIBRATING..."
    calib_color = (0, 255, 0) if status.get('calibration') else (0, 165, 255)
    cv2.putText(vis_frame, f"Calibration: {calib_status} (1px={status.get('px_to_cm_ratio', 0.0):.3f}cm)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, calib_color, 2)
    
    # 2. Рисование результатов
    for person in meta:
        
        # NOTE: Box и keypoints уже в нормированном пространстве (как вернул сервер)
        box = np.array(person.get('box')).reshape(-1, 2) 
        keypoints = np.array(person.get('keypoints'))[:, :2] 
        
        person_id = person.get('id')
        person_status = person.get('status')
        features = person.get('features', {})

        # --- ОБРАТНАЯ ГОМОГРАФИЯ ---
        if HOMOGRAPHY_ENABLED and H_inv is not None:
            box_original_pts = project_points(box, H_inv)
            # Извлекаем x1, y1, x2, y2 из проекции двух углов
            x1, y1 = box_original_pts[0]
            x2, y2 = box_original_pts[1]
            keypoints_original = project_points(keypoints, H_inv)
        else:
            # Используем сырые координаты, если H_inv недоступна (для отладки)
            x1, y1, x2, y2 = [int(i) for i in person.get('box')] 
            keypoints_original = keypoints.astype(int)

        # 3. Рисование Бокса и Скелета
        color = (0, 255, 0)
        if person_status == 'ANOMALY':
            color = (0, 0, 255)
        
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        
        # Скелет
        for i, j in SKELETON_CONNECTIONS:
            p1 = keypoints_original[i]
            p2 = keypoints_original[j]
            # Проверяем, что точки валидны (x > 0)
            if p1[0] > 0 and p2[0] > 0:
                 cv2.line(vis_frame, tuple(p1), tuple(p2), color, 2)

        # Точки
        for x, y in keypoints_original:
            if x > 0 and y > 0:
                cv2.circle(vis_frame, (x, y), 4, (255, 0, 0), -1)

        # Текст (ID, Features)
        status_text = f"ID {person_id} | Status: {person_status}"
        info_text = f"H={features.get('height_cm', 0):.1f}cm, W={features.get('shoulder_width_cm', 0):.1f}cm"

        cv2.putText(vis_frame, status_text, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(vis_frame, info_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return vis_frame

# --- Основная функция ---

def run_video_test():
    """Основной цикл для чтения видео, отправки кадров и отображения результатов."""
    if not os.path.exists(VIDEO_PATH):
        logger.error(f"Видеофайл не найден: {VIDEO_PATH}")
        return

    # 1. ЗАГРУЗКА ГОМОГРАФИИ
    H_matrix = load_homography_matrix(HOMOGRAPHY_MATRIX_PATH)
    H_inv = np.linalg.inv(H_matrix) if HOMOGRAPHY_ENABLED and H_matrix is not None else None

    if HOMOGRAPHY_ENABLED and H_inv is None:
        logger.error("Обратная гомография включена, но матрица H не найдена. Рамки будут неточными!")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"Не удалось открыть видеофайл: {VIDEO_PATH}")
        return

    logger.info(f"Начало обработки видео: {VIDEO_PATH}")
    logger.info("Нажмите 'q' для выхода. Кадр клиента отображает проекцию данных с сервера (с интерполяцией).")
    
    last_anomaly_save_time = time.time()
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- НОВОЕ: Состояние для интерполяции ---
    last_valid_meta: List[Dict] = []
    last_status: Dict = get_api_status() # Загружаем статус до первого кадра
    # ----------------------------------------
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Видеофайл закончился.")
            break
        
        frame_count += 1
        
        # Обработка кадра
        api_result = process_frame(frame.copy()) 
        current_status = get_api_status()
        last_status = current_status # Всегда обновляем текущий статус калибровки
        
        meta_to_draw = []
        anomalies_list = []
        mode_text = "INTERPOLATING"
        mode_color = (0, 165, 255) # Оранжевый

        if api_result:
            if not api_result.get('skipped'):
                # Свежий кадр: обновляем состояние
                last_valid_meta = api_result.get('meta', [])
                meta_to_draw = last_valid_meta
                anomalies_list = api_result.get('anomalies', [])
                mode_text = "PROCESSED"
                mode_color = (0, 255, 0) # Зеленый
            else:
                # Пропущенный кадр: используем последнее известное состояние
                meta_to_draw = last_valid_meta
                # anomalies_list остается пустым, чтобы не сохранять аномалии на каждом кадре

            # 1. Рисование результатов
            frame_to_show = draw_info(frame, meta_to_draw, last_status, H_inv)
            
            # Добавляем режим работы
            cv2.putText(frame_to_show, f"Mode: {mode_text}", 
                        (frame_to_show.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            
            # Сохранение аномалий (только если они были в свежем, непропущенном кадре)
            if anomalies_list and time.time() - last_anomaly_save_time > 1.0:
                logger.warning(f"Anomalies detected: {len(anomalies_list)} in frame {frame_count}")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(ANOMALY_DIR, f"anomaly_{timestamp}_{frame_count}.jpg")
                cv2.imwrite(filename, frame)
                last_anomaly_save_time = time.time()
                
        else:
            # API недоступен, рисуем последний известный статус
            frame_to_show = draw_info(frame, last_valid_meta, last_status, H_inv)
            cv2.putText(frame_to_show, "API ERROR", 
                        (frame_to_show.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # 2. РЕСАЙЗ для отображения клиентского кадра
        if frame_to_show.shape[1] > DISPLAY_MAX_WIDTH:
            scale = DISPLAY_MAX_WIDTH / frame_to_show.shape[1]
            h = int(frame_to_show.shape[0] * scale)
            w = DISPLAY_MAX_WIDTH
            frame_to_show = cv2.resize(frame_to_show, (w, h))

        cv2.imshow('Client View (Original Frame)', frame_to_show)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Тестирование видео завершено.")

if __name__ == "__main__":
    run_video_test()