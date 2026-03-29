# ============================================
# AQI PREDICTOR - CLEAN & SAFE VERSION
# ============================================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# ============================================
# Step 2: Load Dataset
# ============================================

try:
    data = pd.read_csv("aqi_data.csv")
    print("Dataset loaded successfully!\n")
except Exception as e:
    print("Error loading dataset. Check file name/path.\n")
    print(e)
    exit()
# ============================================
# Step 3: Explore Dataset
# ============================================

print("First 5 rows:\n")
print(data.head())

print("\nColumns in dataset:\n")
print(data.columns)

print("\nDataset info:\n")
print(data.info())

print("\nChecking missing values:\n")
print(data.isnull().sum())
# ============================================
# Step 4: Handle Missing Values
# ============================================

data = data.fillna(data.mean(numeric_only=True))
print("\nMissing values handled!\n")
# ============================================
# Step 5: Select Features and Target
# ============================================

features = ['PM2.5', 'PM10', 'NO2']
target = 'AQI'

# Validate columns
for col in features + [target]:
    if col not in data.columns:
        print(f"Column '{col}' not found in dataset.")
        exit()

X = data[features]
y = data[target]

# ============================================
# Step 6: Split Data
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData split completed!")
print("Training data size:", len(X_train))
print("Testing data size:", len(X_test))

# ============================================
# Step 7: Create Model
# ============================================

model = LinearRegression()
print("\nModel created!")
# ============================================
# Step 8: Train Model
# ============================================

model.fit(X_train, y_train)
print("\nModel training completed!")
# ============================================
# Step 9: Predictions
# ============================================

predictions = model.predict(X_test)

print("\nSample Predictions:\n")
for i in range(min(5, len(y_test))):
    print(f"Actual: {y_test.iloc[i]}  |  Predicted: {predictions[i]}")

# ============================================
# Step 10: Evaluation
# ============================================

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print("MSE:", mse)
print("RMSE:", rmse)
print("Accuracy (R^2 score):", model.score(X_test, y_test))
