import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Чтение исходного файла
df = pd.read_csv('data/raw.csv')

# Оставляем только нужные столбцы
df = df[[
    'Manufacturer', 'Model', 'Vehicle_type',
    'Sales_in_thousands', 'Engine_size', 'Horsepower',
    'Curb_weight', 'Fuel_efficiency', 'Price_in_thousands'
]]

# Удаляем строки с пропущенными значениями
df = df.dropna(subset=[
    'Manufacturer', 'Model', 'Vehicle_type',
    'Sales_in_thousands', 'Engine_size', 'Horsepower',
    'Curb_weight', 'Fuel_efficiency', 'Price_in_thousands'
]).reset_index(drop=True)

# Кодируем категориальные столбцы
cat_columns = ['Manufacturer', 'Model', 'Vehicle_type']
encoder = OrdinalEncoder()
df[cat_columns] = encoder.fit_transform(df[cat_columns])

# Сохраняем очищенные данные
df.to_csv('data/clean.csv', index=False)
