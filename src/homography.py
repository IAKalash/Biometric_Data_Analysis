import cv2
import numpy as np
import json
import os
import sys
from loguru import logger
from typing import Optional, Tuple
from config_opt import settings

os.environ["QT_QPA_PLATFORM"] = "xcb" 

INPUT_FILE_PATH = os.path.join("data", "video.webm") 
DISPLAY_MAX_WIDTH = 1000 

TARGET_SIZE: Tuple[int, int] = settings.TARGET_SIZE 
TARGET_W, TARGET_H = TARGET_SIZE
HOMOGRAPHY_FILE = settings.HOMOGRAPHY_MATRIX_PATH 
OFFSET = 0 

TARGET_PTS_COORDS = np.array([
    [OFFSET, OFFSET],
    [TARGET_W - OFFSET, OFFSET],
    [TARGET_W - OFFSET, TARGET_H - OFFSET],
    [OFFSET, TARGET_H - OFFSET]
], dtype=np.float32)

source_points = []
current_source_image = None
original_frame_cache = None
window_name = "Homography Calibration (Source)"

def click_event(event, x, y, flags, param):
    """Сбор 4-х точек на исходном кадре."""
    global source_points, current_source_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(source_points) < 4:
            source_points.append([x, y])

            cv2.circle(current_source_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(current_source_image, str(len(source_points)), (x + 10, y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            logger.info(f"Точка {len(source_points)}/4 выбрана: ({x}, {y})")
 
def calculate_homography():
    """Рассчитывает и сохраняет матрицу гомографии H."""
    global source_points, TARGET_PTS_COORDS, TARGET_SIZE, original_frame_cache

    if len(source_points) != 4:
        logger.error("Ошибка: Выберите ровно 4 исходные точки.")
        return False
    
    src_pts = np.array(source_points, dtype=np.float32)
    dst_pts = TARGET_PTS_COORDS 

    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    if H is None:
        logger.error("Ошибка: Не удалось рассчитать матрицу гомографии.")
        return False

    H_list = H.tolist()
    os.makedirs(os.path.dirname(HOMOGRAPHY_FILE) or '.', exist_ok=True)
    
    with open(HOMOGRAPHY_FILE, 'w') as f:
        json.dump(H_list, f, indent=4)
        
    print("\n" + "=" * 50)
    logger.success(f"Матрица гомографии H успешно рассчитана и сохранена:")
    logger.info(f"   Файл: {HOMOGRAPHY_FILE}")
    logger.info(f"   Размер выходного изображения: {TARGET_SIZE}")
    print("=" * 50)

    if original_frame_cache is None: 
         logger.error("Внутренняя ошибка: Отсутствует оригинальный кадр для проверки.")
         return True
    
    normalized_frame = cv2.warpPerspective(
        original_frame_cache, 
        H, 
        TARGET_SIZE
    )
    
    frame_to_show = normalized_frame.copy()
    if frame_to_show.shape[1] > DISPLAY_MAX_WIDTH:
        scale = DISPLAY_MAX_WIDTH / frame_to_show.shape[1]
        h = int(frame_to_show.shape[0] * scale)
        w = DISPLAY_MAX_WIDTH
        frame_to_show = cv2.resize(frame_to_show, (w, h), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Normalized Output Check", frame_to_show)
    cv2.waitKey(0)
    
    return True

def get_frame_from_input(input_path: str) -> Optional[np.ndarray]:
    """Загружает кадр из изображения или извлекает первый кадр из видео."""
    if not os.path.exists(input_path):
        logger.error(f"Входной файл не найден: {input_path}")
        return None

    # Изображение
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        logger.info(f"Загрузка кадра из изображения: {input_path}")
        return cv2.imread(input_path)

    # Видео
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
        logger.info(f"Загрузка первого кадра из видео: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error(f"Не удалось открыть видеофайл: {input_path}")
            return None

        ret, frame = cap.read()
        cap.release()
        
        if ret:
            logger.info(f"Кадр успешно извлечен из видео. Размер: {frame.shape[1]}x{frame.shape[0]}")
            return frame
        else:
            logger.error("Не удалось прочитать первый кадр из видео.")
            return None
            
    logger.error(f"Неподдерживаемый формат файла: {input_path}")
    return None


if __name__ == "__main__":
    
    input_file_path = INPUT_FILE_PATH

    original_frame = get_frame_from_input(input_file_path)
    
    if original_frame is None:
        sys.exit(1)
        
    current_source_image = original_frame.copy()
    original_frame_cache = original_frame
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)
    
    print("\nИНСТРУКЦИЯ:")
    print("  1. Выберите 4 угла сцены, которую вы хотите выпрямить (например, углы фасада или стен).")
    print("  2. ПОРЯДОК СТРОГО ОБЯЗАТЕЛЕН:")
    print("     P1: Верхний Левый -> P2: Верхний Правый -> P3: Нижний Правый -> P4: Нижний Левый.")
    print("  3. После выбора 4 точек нажмите ENTER для РАСЧЕТА.")
    
    while True:
        temp_img = current_source_image.copy()
        status_text = f"Points: {len(source_points)}/4" if len(source_points) < 4 else "4 points chosen. Press ENTER."
        
        cv2.putText(temp_img, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(window_name, temp_img)
        
        key = cv2.waitKey(10)
        
        if key == 13 and len(source_points) == 4:
            calculate_homography()
            break
        
        if key == 27:
            logger.info("\nПрервано пользователем. Матрица не сохранена.")
            break
            
    cv2.destroyAllWindows()