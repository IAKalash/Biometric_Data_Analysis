# camera_test.py — ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ВЕРСИЯ С АВТОСОХРАНЕНИЕМ
import cv2
import requests
import numpy as np
import time
import os
from datetime import datetime
from loguru import logger
import json

# === Папка для аномалий ===
ANOMALY_DIR = "anomalies"
os.makedirs(ANOMALY_DIR, exist_ok=True)
logger.info(f"Аномалии будут сохраняться в папку: {ANOMALY_DIR}/")

def check_api():
    try:
        r = requests.get("http://localhost:8000/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def process_frame(frame):
    try:
        Drinking = cv2.resize(frame, (640, 480))
        _, img = cv2.imencode('.jpg', Drinking, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        files = {'file': ('frame.jpg', img.tobytes(), 'image/jpeg')}
        r = requests.post("http://localhost:8000/detect", files=files, timeout=7)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug(f"API error: {e}")
    return None

# === Основной цикл ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Не могу открыть камеру")
    exit()

frame_count = 0
real_anomalies = 0
last_send = time.time()
last_result = None  # ← Храним последний результат для отображения

logger.info("Запущен живой тест с сохранением аномалий!")
logger.info("Встань прямо — система откалибруется за 10–15 сек.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    now = time.time()

    # Отправляем кадр раз в 1.5 сек
    if now - last_send >= 1.5:
        last_send = now
        last_result = process_frame(frame)  # ← Сохраняем результат
        
        if last_result and last_result.get("anomalies_detected", 0) > 0:
            real_anomalies += 1
            
            # === Формируем имя файла ===
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            types = "_".join([a['type'].split('_')[-1] for a in last_result.get('anomalies', [])[:2]])
            height = "unknown"
            for person in last_result.get('people_data', []):
                h = person['biometrics'].get('estimated_height_cm')
                if h:
                    height = f"{int(h)}cm"
                    break
            
            filename = f"{ANOMALY_DIR}/{timestamp}_ANOMALY_{types}_{height}_#{real_anomalies}.jpg"
            
            # === Сохраняем кадр и JSON ===
            cv2.imwrite(filename, frame)
            json_path = filename.replace('.jpg', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(last_result, f, indent=2, ensure_ascii=False)

            logger.warning(f"АНОМАЛИЯ #{real_anomalies} СОХРАНЕНА → {os.path.basename(filename)}")
            for a in last_result.get("anomalies", []):
                logger.warning(f"   → {a['type']}: {a['description']} (уверенность: {a['confidence']:.2f})")

    # === Отображаем статус калибровки ===
    if last_result and last_result.get("calibration_status", {}).get("calibrated"):
        status_text = "CALIBRATED"
        status_color = (0, 255, 0)
    else:
        status_text = "CALIBRATING..."
        status_color = (0, 255, 255)

    cv2.putText(frame, f"Anomalies saved: {real_anomalies}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Biometric Anomaly Detection - LIVE", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
logger.info(f"Тест завершён. Всего аномалий сохранено: {real_anomalies} → папка anomalies/")

print("Обучаем ML-модели на синтетических данных...")

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# Генерируем 100 "нормальных" людей (мужчины/женщины)
np.random.seed(42)
n_samples = 100
heights_m = np.random.normal(175, 8, n_samples//2)  # мужчины
shoulders_m = np.random.normal(45, 4, n_samples//2)
heights_f = np.random.normal(162, 7, n_samples//2)  # женщины
shoulders_f = np.random.normal(38, 3, n_samples//2)

# Объединяем
X = np.vstack([
    np.column_stack([heights_m, shoulders_m, shoulders_m/heights_m, np.ones(n_samples//2)]),
    np.column_stack([heights_f, shoulders_f, shoulders_f/heights_f, np.zeros(n_samples//2)])
])

# Обучаем gender classifier
gender_clf = RandomForestClassifier(n_estimators=100, random_state=42)
gender_clf.fit(X[:, :3], X[:, 3])
joblib.dump(gender_clf, 'ml/data/gender_model.pkl')

# Обучаем anomaly detectors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
svm = OneClassSVM(kernel='rbf', nu=0.05)
svm.fit(X_scaled)
joblib.dump(svm, 'ml/data/svm_model.pkl')
forest = IsolationForest(contamination=0.05, random_state=42)
forest.fit(X_scaled)
joblib.dump(forest, 'ml/data/forest_model.pkl')
joblib.dump(scaler, 'ml/data/anomaly_scaler.pkl')

print("✅ Модели обучены и сохранены! Перезапусти сервер.")