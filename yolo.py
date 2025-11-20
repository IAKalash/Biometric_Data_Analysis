from ultralytics import YOLO
import cv2
import numpy as np

# Загружаем модель
model = YOLO("yolov8n-pose.pt")

# Параметры видео
width, height = 640, 416

# Создаём VideoWriter для сохранения видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # кодек для MP4
out = cv2.VideoWriter('output_with_pose.mp4', fourcc, 30.0, (width, height))

print("Начинаю обработку... Нажми 'q' в окне, если захочешь остановить раньше.")

# Создаем окно с правильным размером
cv2.namedWindow("YOLOv8-pose — всё работает!", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8-pose — всё работает!", width, height)

results = model.track(
    source="video.mp4",
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    show=False,
    verbose=False
)

for result in results:
    frame = result.plot()  # уже в RGB

    # Конвертируем RGB в BGR для OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Изменяем размер кадра до нужного размера
    frame_resized = cv2.resize(frame_bgr, (width, height))

    # Показываем в окне
    cv2.imshow("YOLOv8-pose — всё работает!", frame_resized)
    
    # Сохраняем кадр в видео
    out.write(frame_resized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрываем всё
out.release()
cv2.destroyAllWindows()
print("ГОТОВО! Видео сохранено как output_with_pose.mp4 — открывай любым плеером!")