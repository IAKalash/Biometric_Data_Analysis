import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

class TrainingDatasetCreator:
    def __init__(self, csv_path, images_folder):
        self.pose_model = YOLO('yolov8n-pose.pt')
        self.csv_path = csv_path
        self.images_folder = images_folder
        
    def extract_body_features(self, keypoints):
        """Извлекаем числовые признаки из ключевых точек"""
        features = {}
        
        # Базовые размеры (в пикселях)
        features['height'] = self._calculate_height(keypoints)
        features['shoulder_width'] = self._calculate_shoulder_width(keypoints)
        features['hip_width'] = self._calculate_hip_width(keypoints)
        
        # Пропорции
        if features['hip_width'] > 0:
            features['shoulder_hip_ratio'] = features['shoulder_width'] / features['hip_width']
        else:
            features['shoulder_hip_ratio'] = 0
            
        # Дополнительные признаки
        features['torso_length'] = self._calculate_torso_length(keypoints)
        features['leg_length'] = self._calculate_leg_length(keypoints)
        
        if features['leg_length'] > 0:
            features['torso_leg_ratio'] = features['torso_length'] / features['leg_length']
        else:
            features['torso_leg_ratio'] = 0
            
        return features
    
    def _calculate_height(self, keypoints):
        """Рост от носа до лодыжек"""
        if keypoints[0][0] > 0 and keypoints[15][0] > 0:
            return abs(keypoints[15][1] - keypoints[0][1])
        return 0
    
    def _calculate_shoulder_width(self, keypoints):
        """Ширина плеч"""
        if keypoints[5][0] > 0 and keypoints[6][0] > 0:
            return np.linalg.norm(keypoints[5] - keypoints[6])
        return 0
    
    def _calculate_hip_width(self, keypoints):
        """Ширина бедер"""
        if keypoints[11][0] > 0 and keypoints[12][0] > 0:
            return np.linalg.norm(keypoints[11] - keypoints[12])
        return 0
    
    def _calculate_torso_length(self, keypoints):
        """Длина торса (плечи до бедер)"""
        if (keypoints[5][0] > 0 and keypoints[6][0] > 0 and 
            keypoints[11][0] > 0 and keypoints[12][0] > 0):
            shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
            hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
            return abs(hip_y - shoulder_y)
        return 0
    
    def _calculate_leg_length(self, keypoints):
        """Длина ног (бедра до лодыжек)"""
        if keypoints[11][0] > 0 and keypoints[15][0] > 0:
            return abs(keypoints[15][1] - keypoints[11][1])
        return 0
    
    def generate_velocity(self, is_child, is_elderly):
        """Генерируем ТОЛЬКО скорость - остальное берем из CSV"""
        if is_child:
            return round(np.random.uniform(3.0, 7.0), 2)  # дети быстрые
        elif is_elderly:
            return round(np.random.uniform(0.5, 2.5), 2)  # пожилые медленные
        else:
            return round(np.random.uniform(1.5, 5.0), 2)  # взрослые
    
    def create_training_dataset(self):
        """Создаем датасет для обучения"""
        # Читаем исходный CSV
        df_original = pd.read_csv(self.csv_path)
        training_data = []
        
        print(f"Обрабатываем {len(df_original)} изображений...")
        
        for idx, row in df_original.iterrows():
            try:
                image_path = os.path.join(self.images_folder, row['Image'])
                
                if not os.path.exists(image_path):
                    continue
                
                # Загружаем изображение
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Детекция ключевых точек
                results = self.pose_model(img, verbose=False)
                
                if len(results[0].keypoints) == 0:
                    continue
                
                # Извлекаем ключевые точки первого человека
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                
                # Извлекаем признаки тела
                body_features = self.extract_body_features(keypoints)
                
                # БЕРЕМ ВСЕ МЕТКИ НАПРЯМУЮ ИЗ CSV КАК ЕСТЬ!
                training_record = {
                    # ПРИЗНАКИ ИЗ КЛЮЧЕВЫХ ТОЧЕК (числовые)
                    'height': body_features['height'],
                    'shoulder_width': body_features['shoulder_width'],
                    'hip_width': body_features['hip_width'],
                    'shoulder_hip_ratio': body_features['shoulder_hip_ratio'],
                    'torso_length': body_features['torso_length'],
                    'leg_length': body_features['leg_length'],
                    'torso_leg_ratio': body_features['torso_leg_ratio'],
                    
                    # СГЕНЕРИРОВАННАЯ СКОРОСТЬ
                    'velocity': self.generate_velocity(
                        row['AgeLess18'] == 1, 
                        row.get('AgeOver60', 0) == 1
                    ),
                    
                    # МЕТКИ ИЗ CSV (берем как есть - они уже числовые!)
                    'Female': row['Female'],           # 0 или 1
                    'Side': row['Side'],               # 0 или 1  
                    'Front': row['Front'],             # 0 или 1
                    'AgeLess18': row['AgeLess18'],   # 0 или 1
                }
                
                training_data.append(training_record)
                
                if (idx + 1) % 100 == 0:
                    print(f"Обработано {idx + 1} изображений")
                    
            except Exception as e:
                print(f"Ошибка с {row['Image']}: {e}")
                continue
        
        return pd.DataFrame(training_data)

# ОБУЧЕНИЕ МОДЕЛИ
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_gender_classifier(csv_file):
    """Обучаем модель для классификации пола"""
    
    # Загружаем датасет
    df = pd.read_csv(csv_file)
    print(f"Загружено {len(df)} записей")
    
    # ПРИЗНАКИ для обучения
    feature_columns = [
        'height', 'shoulder_width', 'hip_width', 'shoulder_hip_ratio',
        'torso_length', 'leg_length', 'torso_leg_ratio', 'velocity'
    ]
    
    # Убедимся что все признаки есть
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Используем признаки: {available_features}")
    
    # МЕТКИ - Female из CSV (уже 0/1)
    X = df[available_features].values
    y = df['Female'].values  # Уже числовые!
    
    print(f"Размерность: X {X.shape}, y {y.shape}")
    print(f"Распределение классов: {pd.Series(y).value_counts()}")
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Обучаем Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Оценка качества
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Точность модели: {accuracy:.4f}")
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred, target_names=['Male', 'Female']))
    
    # Важность признаков
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nВажность признаков:")
    print(feature_importance)
    
    # Сохраняем модель
    joblib.dump(model, 'gender_classifier.pkl')
    print("Модель сохранена как 'gender_classifier.pkl'")
    
    return model, accuracy

# ИСПОЛЬЗОВАНИЕ
if __name__ == "__main__":
    # 1. Создаем датасет
    CSV_PATH = "archive/PA-100K/test.csv"  # Замени на свой путь
    IMAGES_FOLDER = "archive/PA-100K/data" # Замени на свой путь
    
    creator = TrainingDatasetCreator(CSV_PATH, IMAGES_FOLDER)
    training_df = creator.create_training_dataset()
    
    if len(training_df) > 0:
        # Сохраняем датасет
        training_df.to_csv('training_dataset.csv', index=False)
        print(f"✅ Даатет сохранен: {len(training_df)} записей")
        
        # 2. Обучаем модель
        print("\n🚀 Обучаем модель...")
        model, accuracy = train_gender_classifier('training_dataset.csv')
        
        print(f"\n🎉 Обучение завершено! Точность: {accuracy:.2%}")
    else:
        print("❌ Не удалось создать датасет")