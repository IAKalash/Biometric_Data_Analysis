# database.py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from loguru import logger
from config import settings
from datetime import datetime, timedelta
from typing import Optional

class DatabaseManager:
    def __init__(self) -> None:
        self.engine = create_engine(settings.db.url, echo=settings.db.echo)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def init_db(self) -> None:
        """Initialize database with tables"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS anomaly_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        track_id INTEGER,
                        anomaly_type TEXT,
                        confidence REAL,
                        description TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_timestamp ON anomaly_events(timestamp)"))
            logger.info("Database initialized")
        except Exception as e:
            logger.error(f"Database init error: {e}")
            raise

    def cleanup_old(self, days: int = 30) -> None:
        """Cleanup old records"""
        try:
            with self.SessionLocal() as session:
                cutoff = datetime.now() - timedelta(days=days)
                session.execute(text("""
                    DELETE FROM anomaly_events WHERE timestamp < :cutoff
                """), {'cutoff': cutoff})
                session.commit()
            logger.info(f"Cleaned up records older than {days} days")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Global instance
db_manager = DatabaseManager()