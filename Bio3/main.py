# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2

from service import tracker_service

app = FastAPI(title="BioAnomaly AI Core v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_frame")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Обработка
    result = tracker_service.process_frame(frame)
    
    return result

@app.get("/status")
def status():
    # 🛑 FIX: Возвращаем дополнительные данные для отладки
    # 🛑 FIX: Ключ изменен на 'calibrated', чтобы совпадать с camera_test.py
    return {
        "system": "online",
        "tracked_people": len(tracker_service.states),
        "calibrated": tracker_service.geometry.is_calibrated, 
        "px_to_cm": tracker_service.geometry.px_to_cm_ratio
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)