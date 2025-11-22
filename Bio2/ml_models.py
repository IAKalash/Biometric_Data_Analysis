# ml_models.py
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Optional, Any
from loguru import logger
import os
from config import settings

MODEL_DIR = "ml/data"
os.makedirs(MODEL_DIR, exist_ok=True)

class GenderClassifier:
    """ML gender classifier"""
    def __init__(self) -> None:
        self.model_path = os.path.join(MODEL_DIR, "gender_model.pkl")
        self.scaler_path = os.path.join(MODEL_DIR, "gender_scaler.pkl")
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
        except FileNotFoundError:
            self.model = RandomForestClassifier(n_estimators=settings.ml.n_estimators, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
    
    def predict(self, biometrics: Dict[str, float]) -> Dict[str, Any]:
        """Predict gender with ML or rule-based fallback"""
        height = biometrics.get('estimated_height_cm', 0.0)
        shoulders = biometrics.get('shoulder_width_cm', 0.0)
        
        if height <= 0 or shoulders <= 0:
            return {'gender': 'unknown', 'confidence': 0.0, 'method': 'no_data'}
        
        ratio = shoulders / height
        
        if self.is_trained:
            try:
                features = np.array([[height, shoulders, ratio]])
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled)[0]
                proba = self.model.predict_proba(features_scaled)[0]
                confidence = max(proba)
                return {
                    'gender': 'male' if prediction == 1 else 'female',
                    'confidence': float(confidence),
                    'method': 'ml_model'
                }
            except Exception as e:
                logger.warning(f"ML classifier error: {e}")
        
        # Rule-based fallback
        if shoulders > 48 and height > 170 and ratio > 0.28:
            return {'gender': 'male', 'confidence': 0.7, 'method': 'rule_based'}
        elif shoulders < 42 and height < 175 and ratio < 0.30:
            return {'gender': 'female', 'confidence': 0.7, 'method': 'rule_based'}
        else:
            return {'gender': 'unknown', 'confidence': 0.5, 'method': 'rule_based'}
    
    def save(self) -> None:
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

class MLAnomalyDetector:
    """ML anomaly detector"""
    def __init__(self) -> None:
        self.model_path_svm = os.path.join(MODEL_DIR, "svm_model.pkl")
        self.model_path_forest = os.path.join(MODEL_DIR, "forest_model.pkl")
        self.scaler_path = os.path.join(MODEL_DIR, "anomaly_scaler.pkl")
        try:
            self.svm = joblib.load(self.model_path_svm)
            self.forest = joblib.load(self.model_path_forest)
            self.scaler = joblib.load(self.scaler_path)
            self.is_trained = True
        except FileNotFoundError:
            self.svm = OneClassSVM(kernel='rbf', nu=0.1)
            self.forest = IsolationForest(contamination=0.1, random_state=42)
            self.scaler = StandardScaler()
            self.is_trained = False
        self.normal_samples: List[List[float]] = []
    
    def detect_anomalies(self, biometrics: Dict[str, float], gender: str) -> List[Dict[str, Any]]:
        """Detect ML anomalies"""
        if not self.is_trained or len(self.normal_samples) < settings.ml.min_samples:
            return []
        
        height = biometrics.get('estimated_height_cm', 0.0)
        shoulders = biometrics.get('shoulder_width_cm', 0.0)
        
        if height <= 0 or shoulders <= 0:
            return []
        
        ratio = shoulders / height
        gender_encoded = 1.0 if gender == 'male' else 0.0 if gender == 'female' else 0.5
        
        try:
            features = np.array([[height, shoulders, ratio, gender_encoded]])
            features_scaled = self.scaler.transform(features)
            
            anomalies = []
            svm_pred = self.svm.predict(features_scaled)[0]
            forest_pred = self.forest.predict(features_scaled)[0]
            
            if svm_pred == -1 and forest_pred == -1:  # Ensemble
                anomalies.append({
                    'type': 'ml_anomaly',
                    'confidence': 0.8,
                    'description': 'ML: statistical anomaly (ensemble)',
                    'severity': 'medium'
                })
            elif svm_pred == -1:
                anomalies.append({
                    'type': 'ml_svm_anomaly',
                    'confidence': 0.8,
                    'description': 'ML: SVM anomaly',
                    'severity': 'medium'
                })
            elif forest_pred == -1:
                anomalies.append({
                    'type': 'ml_forest_anomaly',
                    'confidence': 0.7,
                    'description': 'ML: Forest anomaly',
                    'severity': 'medium'
                })
            
            return anomalies
        except Exception as e:
            logger.error(f"ML detection error: {e}")
            return []
    
    def add_normal_sample(self, biometrics: Dict[str, float], gender: str) -> None:
        """Add normal sample for training"""
        height = biometrics.get('estimated_height_cm', 0.0)
        shoulders = biometrics.get('shoulder_width_cm', 0.0)
        
        if height > 0 and shoulders > 0:
            ratio = shoulders / height
            gender_encoded = 1.0 if gender == 'male' else 0.0 if gender == 'female' else 0.5
            self.normal_samples.append([height, shoulders, ratio, gender_encoded])
    
    def train_models(self) -> None:
        """Train ML models on collected data"""
        if len(self.normal_samples) < settings.ml.min_samples:
            return
        
        try:
            X = np.array(self.normal_samples)
            X_scaled = self.scaler.fit_transform(X)
            
            self.svm.fit(X_scaled)
            self.forest.fit(X_scaled)
            scores = cross_val_score(self.forest, X_scaled, cv=3)
            logger.info(f"Cross-val scores: {scores.mean()}")
            self.is_trained = True
            self.save()
            
            logger.info(f"ML models trained on {len(X)} samples")
        except Exception as e:
            logger.error(f"ML training error: {e}")
    
    def save(self) -> None:
        joblib.dump(self.svm, self.model_path_svm)
        joblib.dump(self.forest, self.model_path_forest)
        joblib.dump(self.scaler, self.scaler_path)

class MLModels:
    """Main class for ML models management"""
    def __init__(self) -> None:
        self.gender_classifier = GenderClassifier()
        self.anomaly_detector = MLAnomalyDetector()
        logger.info("ML models initialized")
    
    def process_person(self, biometrics: Dict[str, float]) -> Dict[str, Any]:
        """Process person with all ML models"""
        gender_result = self.gender_classifier.predict(biometrics)
        
        ml_anomalies = self.anomaly_detector.detect_anomalies(biometrics, gender_result['gender'])
        
        if gender_result['confidence'] > 0.7 and biometrics.get('estimated_height_cm', 0) > 100:
            self.anomaly_detector.add_normal_sample(biometrics, gender_result['gender'])
        
        if len(self.anomaly_detector.normal_samples) % 20 == 0:
            self.anomaly_detector.train_models()
        
        return {
            'gender': gender_result,
            'ml_anomalies': ml_anomalies
        }