import cv2
import numpy as np
import json
import os
import sys

# --- ФИКС GUI для Linux (если вы используете удаленное подключение)
os.environ["QT_QPA_PLATFORM"] = "xcb" 

# ================= КОНФИГУРАЦИЯ СКРИПТА =================
CAMERA_NAME = "cam_01"          
DATA_DIR = "data"               
MODELS_DIR = "models"           
TEST_FRAME_NAME = "test2.jpg" 

# Параметры целевого (нормированного) изображения
TARGET_SIZE = (1500, 1000) # Квадрат для сохранения пропорций
TARGET_W, TARGET_H = TARGET_SIZE
OFFSET = 50

# Целевые точки (Destination Points) - АВТОМАТИЧЕСКИЙ РАСЧЕТ
# Создаем идеальный квадрат (с отступами) для выпрямленной сцены.
TARGET_PTS_COORDS = np.array([
    [OFFSET, OFFSET],                       # P'1: Верхний левый угол
    [TARGET_W - OFFSET, OFFSET],            # P'2: Верхний правый угол
    [TARGET_W - OFFSET, TARGET_H - OFFSET], # P'3: Нижний правый угол
    [OFFSET, TARGET_H - OFFSET]             # P'4: Нижний левый угол
], dtype=np.float32)

# Полные пути
TEST_FRAME_PATH = os.path.join(DATA_DIR, TEST_FRAME_NAME)
HOMOGRAPHY_FILE = os.path.join(MODELS_DIR, f"H_{CAMERA_NAME}.json")

# Глобальные переменные для хранения точек
source_points = []
current_source_image = None
window_name = "Homography Calibration (Source)"

# ================= ФУНКЦИИ =================

def click_event(event, x, y, flags, param):
    """Обработчик событий мыши для сбора ИСХОДНЫХ точек (Source Points)."""
    global source_points, current_source_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(source_points) < 4:
            # 1. Добавляем точку
            source_points.append([x, y])
            
            # 2. Отрисовываем точку на изображении
            cv2.circle(current_source_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(current_source_image, str(len(source_points)), (x + 10, y + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 3. Обновляем окно
            print(f"Точка {len(source_points)}/4 выбрана: ({x}, {y})")
            
def calculate_and_save_homography():
    """Рассчитывает матрицу H и сохраняет ее в JSON, используя АВТОМАТИЧЕСКИЕ целевые точки."""
    global source_points, TARGET_PTS_COORDS, TARGET_SIZE

    if len(source_points) != 4:
        print("❌ Ошибка: Выберите ровно 4 исходные точки.")
        return False
    
    src_pts = np.array(source_points, dtype=np.float32)
    dst_pts = TARGET_PTS_COORDS 

    # Расчет матрицы гомографии
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    if H is None:
        print("❌ Ошибка: Не удалось рассчитать матрицу гомографии. Точки могут быть коллинеарными.")
        return False

    # Сохранение матрицы в JSON
    H_list = H.tolist()
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    with open(HOMOGRAPHY_FILE, 'w') as f:
        json.dump(H_list, f, indent=4)
        
    print("\n" + "=" * 50)
    print(f"✅ Матрица гомографии H успешно рассчитана и сохранена:")
    print(f"   Файл: {HOMOGRAPHY_FILE}")
    print(f"   Размер выходного изображения: {TARGET_SIZE}")
    print("=" * 50)
    
    # Визуализация результата
    print("Визуализирую результат преобразования (Normalized Output Check)...")
    original_frame = cv2.imread(TEST_FRAME_PATH)
    normalized_frame = cv2.warpPerspective(
        original_frame, 
        H, 
        TARGET_SIZE
    )
    cv2.imshow("Normalized Output Check (Final Result)", normalized_frame)
    cv2.waitKey(0)
    
    return True


# ================= ОСНОВНАЯ ЛОГИКА =================
if __name__ == "__main__":
    
    # 1. Проверка и инициализация
    if not os.path.exists(TEST_FRAME_PATH):
        print(f"[ERROR] Не найден тестовый кадр: {TEST_FRAME_PATH}")
        print(f"Пожалуйста, поместите {TEST_FRAME_NAME} в папку '{DATA_DIR}' и перезапустите.")
        sys.exit(1)

    original_frame = cv2.imread(TEST_FRAME_PATH)
    if original_frame is None:
        print("[ERROR] Не удалось загрузить изображение.")
        sys.exit(1)
        
    current_source_image = original_frame.copy()
    
    # 2. Настройка окна и обработчика
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, click_event)
    
    print("\n--- ЗАПУСК КАЛИБРОВКИ ГОМОГРАФИИ (ВИД СБОКУ/ФАС) ---")
    print("ИНСТРУКЦИЯ (ОДИН ЭТАП):")
    print("  1. Выберите 4 угла ТРЕХМЕРНОЙ СЦЕНЫ, которую вы хотите выпрямить (например, углы фасада или стен).")
    print("  2. ПОРЯДОК СТРОГО ОБЯЗАТЕЛЕН:")
    print("     P1: Верхний Левый -> P2: Верхний Правый -> P3: Нижний Правый -> P4: Нижний Левый.")
    print("  3. После выбора 4 точек нажмите ENTER для РАСЧЕТА.")
    
    # 3. Основной цикл
    while True:
        # Обновляем изображение с подсказкой
        temp_img = current_source_image.copy()
        status_text = f"Кликните {len(source_points)+1}/4 точки. Нажмите ENTER для расчета." if len(source_points) < 4 else "4 точки выбраны. Нажмите ENTER."
        
        cv2.putText(temp_img, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(window_name, temp_img)
        
        key = cv2.waitKey(10)
        
        if key == 13 and len(source_points) == 4: # ENTER
            calculate_and_save_homography()
            break
        
        if key == 27: # ESC
            print("\nПрервано пользователем. Матрица не сохранена.")
            break
            
    cv2.destroyAllWindows()