import cv2
import numpy as np
import json
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "xcb" 

CAMERA_NAME = "cam_01"          
DATA_DIR = "data"               
MODELS_DIR = "models"           
TEST_FRAME_NAME = "image.png" 

# Параметры целевого изображения
TARGET_SIZE = (1500, 1000)
TARGET_W, TARGET_H = TARGET_SIZE
OFFSET = 0


TARGET_PTS_COORDS = np.array([
    [OFFSET, OFFSET],
    [TARGET_W - OFFSET, OFFSET],
    [TARGET_W - OFFSET, TARGET_H - OFFSET],
    [OFFSET, TARGET_H - OFFSET]
], dtype=np.float32)

TEST_FRAME_PATH = os.path.join(DATA_DIR, TEST_FRAME_NAME)
HOMOGRAPHY_FILE = os.path.join(MODELS_DIR, f"H_{CAMERA_NAME}.json")

source_points = []
current_source_image = None
window_name = "Homography Calibration (Source)"

# Обработчик сбора точек на изображении
def click_event(event, x, y, flags, param):
    global source_points, current_source_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(source_points) < 4:
            source_points.append([x, y])

            cv2.circle(current_source_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(current_source_image, str(len(source_points)), (x + 10, y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            print(f"Точка {len(source_points)}/4 выбрана: ({x}, {y})")

# Рассчитывает матрицу H и сохраняет ее в JSON     
def calculate_homography():
    global source_points, TARGET_PTS_COORDS, TARGET_SIZE

    if len(source_points) != 4:
        print("Ошибка: Выберите ровно 4 исходные точки.")
        return False
    
    src_pts = np.array(source_points, dtype=np.float32)
    dst_pts = TARGET_PTS_COORDS 

    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    if H is None:
        print("Ошибка: Не удалось рассчитать матрицу гомографии.")
        return False

    H_list = H.tolist()
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(HOMOGRAPHY_FILE, 'w') as f:
        json.dump(H_list, f, indent=4)
        
    print("\n" + "=" * 50)
    print(f"Матрица гомографии H успешно рассчитана и сохранена:")
    print(f"   Файл: {HOMOGRAPHY_FILE}")
    print(f"   Размер выходного изображения: {TARGET_SIZE}")
    print("=" * 50)

    original_frame = cv2.imread(TEST_FRAME_PATH)
    normalized_frame = cv2.warpPerspective(
        original_frame, 
        H, 
        TARGET_SIZE
    )
    cv2.imshow("Normalized Output Check", normalized_frame)
    cv2.waitKey(0)
    
    return True


if __name__ == "__main__":
    
    if not os.path.exists(TEST_FRAME_PATH):
        print(f"[ERROR] Не найден тестовый кадр: {TEST_FRAME_PATH}")
        print(f"Пожалуйста, поместите {TEST_FRAME_NAME} в папку '{DATA_DIR}' и перезапустите.")
        sys.exit(1)

    original_frame = cv2.imread(TEST_FRAME_PATH)
    if original_frame is None:
        print("[ERROR] Не удалось загрузить изображение.")
        sys.exit(1)
        
    current_source_image = original_frame.copy()
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)
    
    print("ИНСТРУКЦИЯ:")
    print("  1. Выберите 4 угла сцены, которую вы хотите выпрямить (например, углы фасада или стен).")
    print("  2. ПОРЯДОК СТРОГО ОБЯЗАТЕЛЕН:")
    print("     P1: Верхний Левый -> P2: Верхний Правый -> P3: Нижний Правый -> P4: Нижний Левый.")
    print("  3. После выбора 4 точек нажмите ENTER для РАСЧЕТА.")
    
    while True:
        temp_img = current_source_image.copy()
        status_text = f"{len(source_points)}/4 points." if len(source_points) < 4 else "4 points chosen. Press ENTER."
        
        cv2.putText(temp_img, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(window_name, temp_img)
        
        key = cv2.waitKey(10)
        
        if key == 13 and len(source_points) == 4:
            calculate_homography()
            break
        
        if key == 27:
            print("\nПрервано пользователем. Матрица не сохранена.")
            break
            
    cv2.destroyAllWindows()