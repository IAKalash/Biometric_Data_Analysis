import cv2
import os
import sys
import numpy as np
import json
from ultralytics import YOLO

os.environ["QT_QPA_PLATFORM"] = "xcb"

CAMERA_NAME = "cam_01"
SAVE_VIDEO = False

# Настройки модели
BASE_MODEL_NAME = "yolov8n-pose.pt"
INFERENCE_SIZE = 960       # Разрешение
CONF_THRESHOLD = 0.3       # Порог уверенности
SKIP_FRAMES = 3            # Модель работает каждый n+1 кадр

# Настройки Гомографии
HOMOGRAPHY_ENABLED = True
TARGET_SIZE = (1500, 1000)

# Настройка путей
VIDEO_FILENAME = "video.webm"
MODELS_DIR = "models"
RESULTS_DIR = "results"
DATA_DIR = "data"

HOMOGRAPHY_MATRIX_FILE = os.path.join(MODELS_DIR, f"H_{CAMERA_NAME}.json")
VIDEO_PATH = os.path.join(DATA_DIR, VIDEO_FILENAME)
OUTPUT_VIDEO_PATH = os.path.join(RESULTS_DIR, "output.mp4") 
DATA_OUTPUT_PATH = os.path.join(RESULTS_DIR, "biometric_data.json")

SKELETON_CONNECTIONS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), 
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), 
    (0, 5), (0, 6)
]
LINE_COLOR = (255, 0, 255)
POINT_COLOR = (0, 255, 255)

# Загружает матрицу гомографии H из JSON файла.
def load_homography_matrix(path):
    if not os.path.exists(path):
        print(f"[ERROR] Матрица гомографии не найдена: {path}")
        print("Пожалуйста, создайте ее с помощью homography.py")
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

# Применяет гомографию к кадру.
def normalize_frame(frame, H, target_size):
    normalized_frame = cv2.warpPerspective(
        frame, 
        H, 
        target_size
    )
    return normalized_frame

# Выбирает или создает ONNX модель под нужное разрешение.
def get_optimized_model():
    onnx_filename = os.path.join(MODELS_DIR, f"{BASE_MODEL_NAME.replace('.pt', '')}-{INFERENCE_SIZE}.onnx")
    
    if not os.path.exists(onnx_filename):
        print(f"[INFO] Создаю модель {onnx_filename}...")
        try:
            model = YOLO(os.path.join(MODELS_DIR, BASE_MODEL_NAME))
            model.export(format="onnx", imgsz=INFERENCE_SIZE, dynamic=False, simplify=True)
            default_onnx = os.path.join(MODELS_DIR, BASE_MODEL_NAME.replace(".pt", ".onnx"))
            os.rename(default_onnx, onnx_filename)

            print(f"[SUCCESS] Модель готова: {onnx_filename}")
        except Exception as e:
            print(f"[ERROR] Ошибка экспорта: {e}")
            sys.exit(1)
    else:
        print(f"[INFO] Найдена готовая модель: {onnx_filename}")

    return YOLO(onnx_filename, task='pose')

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Не найден исходный файл видео: {VIDEO_PATH}")
        print(f"Пожалуйста, поместите 'video.mp4' в папку '{DATA_DIR}' и перезапустите.")
        sys.exit(1)

    model = get_optimized_model()
    
    H_matrix = None
    if HOMOGRAPHY_ENABLED:
        H_matrix = load_homography_matrix(HOMOGRAPHY_MATRIX_FILE)
        if H_matrix is None:
            sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_width, output_height = TARGET_SIZE if HOMOGRAPHY_ENABLED else (width, height)
    
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
            
            if HOMOGRAPHY_ENABLED:
                processed_frame = normalize_frame(frame, H_matrix, TARGET_SIZE)
            else:
                processed_frame = frame

            should_process = (frame_idx % (SKIP_FRAMES + 1) == 0)
            
            if should_process:
                results = model.track(
                    processed_frame,
                    persist=True, 
                    tracker="bytetrack.yaml",
                    imgsz=INFERENCE_SIZE, 
                    conf=CONF_THRESHOLD,
                    verbose=False
                )
                
                cached_skeletons = []
                current_frame_data = []

                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    keypoints = results[0].keypoints.xy.cpu().numpy()
                    
                    for pid, box, kp in zip(ids, boxes, keypoints):
                        cached_skeletons.append({"id": pid, "box": box, "kp": kp})
                        
                        current_frame_data.append({
                            "person_id": int(pid),
                            "frame_idx": frame_idx,
                            "keypoints": kp.tolist(), 
                            "bbox": box.tolist()
                        })
                
                all_biometric_data.extend(current_frame_data)
            
            vis_frame = processed_frame.copy()
            
            for person in cached_skeletons:
                box = person["box"]
                kp = person["kp"]
                pid = person["id"]
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                for start_idx, end_idx in SKELETON_CONNECTIONS:
                    p1 = kp[start_idx]
                    p2 = kp[end_idx]
                    if p1[0] > 0 and p2[0] > 0:
                         cv2.line(vis_frame, (int(p1[0]), int(p1[1])), 
                                  (int(p2[0]), int(p2[1])), LINE_COLOR, 2)
                
                for x, y in kp:
                    if x > 0:
                        cv2.circle(vis_frame, (int(x), int(y)), 4, POINT_COLOR, -1)

                cv2.putText(vis_frame, f"ID:{pid}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            mode_text = f"Mode: {'INFERENCE' if should_process else 'INTERPOLATION'} (Norm={HOMOGRAPHY_ENABLED})"
            color = (0, 255, 0) if should_process else (0, 165, 255)
            cv2.putText(vis_frame, mode_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

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
        
        if all_biometric_data:
            with open(DATA_OUTPUT_PATH, 'w') as f:
                json.dump(all_biometric_data, f, indent=4)
            print(f"[INFO] Извлеченные данные сохранены в: {DATA_OUTPUT_PATH}")