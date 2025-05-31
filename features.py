import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
import joblib
import os
# Читаем очищенные данные
df = pd.read_csv('data/clean.csv')
# Отделяем целевую переменную
X = df.drop(columns=['Price_in_thousands'])
y = df['Price_in_thousands']
# Масштабируем X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Нормализуем y
power_trans = PowerTransformer()
y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
os.makedirs("data/features", exist_ok=True)
np.save("data/features/X.npy", X_scaled)
np.save("data/features/y.npy", y_scaled)
joblib.dump(power_trans, "data/features/power_transform.joblib")
