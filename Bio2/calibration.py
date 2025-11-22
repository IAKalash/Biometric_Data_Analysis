# calibration.py
from real_detection import real_detector
import cv2
import time
from loguru import logger
from config import settings
from typing import Optional

def quick_calibration(gender: Optional[str] = None) -> None:
    """Quick system calibration"""
    assumed_height = 170 if gender == 'male' else 160 if gender == 'female' else 170
    logger.info("Starting quick calibration...")
    logger.info("Stand in front of the camera at normal distance")
    
    cap = cv2.VideoCapture(0)
    try:
        samples_collected = 0
        for i in range(30):  # 30 frames for calibration
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Draw instructions on frame
            cv2.putText(frame, "Stand straight for calibration", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Calibration', frame)
            
            result = real_detector.analyze_frame(frame)
            current_samples = result['calibration_status']['samples_collected']
            
            if current_samples > samples_collected:
                samples_collected = current_samples
                logger.info(f"Collected samples: {samples_collected}")
            
            if samples_collected >= settings.ml.min_samples and not real_detector.is_calibrated:
                real_detector.force_calibration(assumed_height)
                break
            
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    logger.info("Calibration completed!")

if __name__ == "__main__":
    quick_calibration()