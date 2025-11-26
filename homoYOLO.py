import cv2
import os
import sys
import numpy as np
import json
from ultralytics import YOLO

# --- ФИКС GUI для Linux ---
# Принудительно используем XCB для совместимости с OpenCV на многих Linux-системах
os.environ["QT_QPA_PLATFORM"] = "xcb"

# ================= КОНФИГУРАЦИЯ =================
MODELS_DIR = "models"
RESULTS_DIR = "results"
DATA_DIR = "data"
VIDEO_FILENAME = "video.webm"
CAMERA_NAME = "cam_01"     # <--- ИМЯ КАМЕРЫ (для загрузки матрицы H)

# Полные пути
VIDEO_PATH = os.path.join(DATA_DIR, VIDEO_FILENAME)
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, "output_analyzed_smooth_norm.mp4")
SAVE_VIDEO = True 
DATA_OUTPUT_PATH = os.path.join(RESULTS_DIR, "biometric_data.json")

# Настройки модели
BASE_MODEL_NAME = "yolov8n-pose.pt"
INFERENCE_SIZE = 960       # Разрешение
CONF_THRESHOLD = 0.4       # Порог уверенности
SKIP_FRAMES = 3            # Модель работает каждый n+1 кадр

# Настройки Гомографии
HOMOGRAPHY_ENABLED = True                 # <--- АКТИВИРОВАТЬ/ДЕЗАКТИВИРОВАТЬ НОРМАЛИЗАЦИЮ
HOMOGRAPHY_MATRIX_FILE = os.path.join(MODELS_DIR, f"H_{CAMERA_NAME}.json")
TARGET_SIZE = (960, 960)                 # Размер нормированного кадра (квадрат)

# Карта соединений для скелета COCO (индексы ключевых точек)
SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), 
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), 
    (0, 5), (0, 6)
]
LINE_COLOR = (255, 0, 255) # Фиолетовый
POINT_COLOR = (0, 255, 255) # Желтый

# ================= ФУНКЦИИ ГОМОГРАФИИ =================
def load_homography_matrix(path):
    """Загружает матрицу гомографии H из JSON файла."""
    if not os.path.exists(path):
        print(f"[ERROR] Матрица гомографии не найдена: {path}")
        print("Пожалуйста, создайте ее с помощью cv2.findHomography и сохраните.")
        return None
    try:
        with open(path, 'r') as f:
            H_list = json.load(f)
        H = np.array(H_list, dtype=np.float64)
        print(f"[INFO] Матрица гомографии H загружена из: {path}")
        return H
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки или форматирования матрицы H: {e}")
        return None

def normalize_frame(frame, H, target_size):
    """Применяет перспективное преобразование (гомографию) к кадру."""
    # Используем cv2.warpPerspective для устранения искажений
    normalized_frame = cv2.warpPerspective(
        frame, 
        H, 
        target_size
    )
    return normalized_frame

# ================= МЕНЕДЖЕР МОДЕЛИ (ONNX) =================
def get_optimized_model():
    """Автоматически выбирает или создает ONNX модель под нужное разрешение."""
    onnx_filename = os.path.join(MODELS_DIR, f"{BASE_MODEL_NAME.replace('.pt', '')}-{INFERENCE_SIZE}.onnx")
    
    # ... (Остальная часть функции get_optimized_model остается без изменений)
    if not os.path.exists(onnx_filename):
        print(f"[INFO] Создаю модель {onnx_filename} (это будет один раз)...")
        try:
            model = YOLO(BASE_MODEL_NAME)
            model.export(format="onnx", imgsz=INFERENCE_SIZE, dynamic=False, simplify=True)
            
            # Перемещаем и переименовываем
            default_onnx = BASE_MODEL_NAME.replace(".pt", ".onnx")
            os.rename(default_onnx, onnx_filename)

            print(f"[SUCCESS] Модель готова: {onnx_filename}")
        except Exception as e:
            print(f"[ERROR] Ошибка экспорта: {e}")
            sys.exit(1)
    else:
        print(f"[INFO] Найдена готовая модель: {onnx_filename}")

    return YOLO(onnx_filename, task='pose')

# ================= ОСНОВНОЙ КОД =================
if __name__ == "__main__":
    # Создание структуры папок
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Не найден исходный файл видео: {VIDEO_PATH}")
        print(f"Пожалуйста, поместите 'video.mp4' в папку '{DATA_DIR}' и перезапустите.")
        sys.exit(1)

    # 1. Загрузка
    model = get_optimized_model()
    
    # --- ЗАГРУЗКА ГОМОГРАФИИ ---
    H_matrix = None
    if HOMOGRAPHY_ENABLED:
        H_matrix = load_homography_matrix(HOMOGRAPHY_MATRIX_FILE)
        if H_matrix is None:
            sys.exit(1) # Выходим, если гомография включена, но матрица не найдена

    # 2. Видеозахват
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Определяем размер выходного видео
    output_width, output_height = TARGET_SIZE if HOMOGRAPHY_ENABLED else (width, height)
    
    # 3. Настройка записи
    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (output_width, output_height))

    cv2.namedWindow("Biometric Analytics - Smooth", cv2.WINDOW_NORMAL)

    frame_idx = 0
    cached_skeletons = [] 
    all_biometric_data = [] 

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break 
            
            frame_idx += 1
            
            # --- НОРМАЛИЗАЦИЯ ПЕРСПЕКТИВЫ (Гомография) ---
            if HOMOGRAPHY_ENABLED and H_matrix is not None:
                # Преобразуем исходный кадр
                processed_frame = normalize_frame(frame, H_matrix, TARGET_SIZE)
            else:
                processed_frame = frame.copy()

            # --- ЛОГИКА ИНФЕРЕНСА ---
            should_process = (frame_idx % (SKIP_FRAMES + 1) == 0)
            
            if should_process:
                # Тяжелая операция: Запускаем нейросеть на обработанном кадре
                results = model.track(
                    processed_frame, # <-- Используем нормированный кадр
                    persist=True, 
                    tracker="bytetrack.yaml",
                    imgsz=INFERENCE_SIZE, 
                    conf=CONF_THRESHOLD,
                    verbose=False
                )
                
                # Обновляем кэш свежими данными и собираем данные для JSON
                cached_skeletons = []
                current_frame_data = []

                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    keypoints = results[0].keypoints.xy.cpu().numpy()
                    
                    for pid, box, kp in zip(ids, boxes, keypoints):
                        # 1. Для кэша (визуализация)
                        cached_skeletons.append({"id": pid, "box": box, "kp": kp})
                        
                        # 2. Для JSON (анализ)
                        current_frame_data.append({
                            "person_id": int(pid),
                            "frame_idx": frame_idx,
                            # Ключевые точки и боксы теперь нормированы!
                            "keypoints": kp.tolist(), 
                            "bbox": box.tolist()
                        })
                
                # Добавляем данные текущего кадра в общий список
                all_biometric_data.extend(current_frame_data)
            
            # --- ОТРИСОВКА (НА КАЖДОМ КАДРЕ) ---
            vis_frame = processed_frame.copy() # <-- Визуализируем нормированный кадр
            
            # Рисуем ПОСЛЕДНИЕ ИЗВЕСТНЫЕ данные на ТЕКУЩЕМ кадре
            for person in cached_skeletons:
                box = person["box"]
                kp = person["kp"]
                pid = person["id"]
                
                # 1. Бокс
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 2. Соединения скелета (линии)
                for start_idx, end_idx in SKELETON_CONNECTIONS:
                    p1 = kp[start_idx]
                    p2 = kp[end_idx]
                    if p1[0] > 0 and p2[0] > 0:
                         cv2.line(vis_frame, (int(p1[0]), int(p1[1])), 
                                  (int(p2[0]), int(p2[1])), LINE_COLOR, 2)
                
                # 3. Ключевые точки (круги)
                for x, y in kp:
                    if x > 0: # Если точка найдена
                        cv2.circle(vis_frame, (int(x), int(y)), 4, POINT_COLOR, -1)

                # 4. ID над головой
                cv2.putText(vis_frame, f"ID:{pid}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Индикация режима
            mode_text = f"Mode: {'INFERENCE' if should_process else 'INTERPOLATION'} (Norm={HOMOGRAPHY_ENABLED})"
            color = (0, 255, 0) if should_process else (0, 165, 255)
            cv2.putText(vis_frame, mode_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # 4. ЗАПИСЬ И ПОКАЗ
            if writer:
                 writer.write(vis_frame) 
            cv2.imshow("Biometric Analytics - Smooth", vis_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Прервано пользователем")
        
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"[INFO] Видео сохранено в: {OUTPUT_VIDEO_PATH}")

        cv2.destroyAllWindows()
        
        # Сохранение извлеченных данных
        if all_biometric_data:
            with open(DATA_OUTPUT_PATH, 'w') as f:
                json.dump(all_biometric_data, f, indent=4)
            print(f"[INFO] Извлеченные данные сохранены в: {DATA_OUTPUT_PATH}")