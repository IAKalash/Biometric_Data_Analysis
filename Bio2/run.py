# run.py
import sys
import os
from argparse import ArgumentParser
from loguru import logger

# Добавляем корневую папку в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    parser = ArgumentParser(description="Biometric Anomaly Detection System")
    parser.add_argument('--workers', type=int, default=1, help="Number of uvicorn workers")
    parser.add_argument('--no-reload', action='store_true', help="Disable auto-reload")
    args = parser.parse_args()

    logger.info("Starting anomaly detection system...")
    logger.info("API: http://0.0.0.0:8000")
    logger.info("Health: http://0.0.0.0:8000/health")
    logger.info("Press Ctrl+C to stop")

    import uvicorn

    # ← ВОТ ЭТО ГЛАВНОЕ ИСПРАВЛЕНИЕ:
    uvicorn.run(
        "main:app",                     # ← как строка, а не объект!
        host="0.0.0.0",
        port=8000,
        reload=not args.no_reload,      # reload по умолчанию включён
        workers=args.workers if args.workers > 1 else None
    )