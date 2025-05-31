import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# Загружаем подготовленные данные
X = np.load("data/features/X.npy")
y = np.load("data/features/y.npy")

# Делим на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучаем модель
model = SGDRegressor(random_state=42)
model.fit(X_train, y_train.ravel())

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
