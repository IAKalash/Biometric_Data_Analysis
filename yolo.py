from ultralytics import YOLO
import cv2
import imageio_ffmpeg as ffmpeg
import numpy as np

# Загружаем модель
model = YOLO("yolov8n-pose.pt")   # или yolov11n-pose.pt

# Параметры твоего видео (посмотри в терминале, какие размеры выводит YOLO)
# У тебя было: 416x640 → меняем на (ширина, высота) = (640, 416)
width, height = 640, 416

# Создаём писатель через ffmpeg (работает ВЕЗДЕ)
writer = ffmpeg.get_writer(
    'output_with_pose.mp4',
    fps=30,
    codec='libx264',           # самый совместимый кодек
    pixel_format='yuv420p',    # важен для плееров и браузеров
    output_params=['-preset', 'fast']
)

print("Начинаю обработку... Нажми 'q' в окне, если захочешь остановить раньше.")

results = model.track(
    source="video.webm",
    stream=True,
    persist=True,
    tracker="bytetrack.yaml",
    show=False,
    verbose=False
)

for result in results:
    frame = result.plot()  # уже в RGB

    # OpenCV работает в BGR, а ffmpeg хочет RGB → конвертируем
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Показываем (если хочешь видеть онлайн)
    cv2.imshow("YOLOv8-pose — всё работает!", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Записываем кадр
    writer.write_frame(frame_bgr)

# Закрываем всё
writer.close()
cv2.destroyAllWindows()
print("ГОТОВО! Видео сохранено как output_with_pose.mp4 — открывай любым плеером!")