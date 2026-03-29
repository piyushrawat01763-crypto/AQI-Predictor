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

# ============================================
# Step 11: Visualization
# ============================================

plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()

# ============================================
# Step 12: User Input Prediction
# ============================================

print("\n--- AQI Prediction System ---")

try:
    pm25 = float(input("Enter PM2.5 value: "))
    pm10 = float(input("Enter PM10 value: "))
    no2 = float(input("Enter NO2 value: "))

    user_data = pd.DataFrame([[pm25, pm10, no2]], columns=features)

    predicted_aqi = model.predict(user_data)
    print("\nPredicted AQI:", predicted_aqi[0])

except Exception as e:
    print("Invalid input! Please enter numbers only.")
    print(e)

# ============================================
# Step 13: Save Model
# ============================================

try:
    joblib.dump(model, "aqi_model.pkl")
    print("\nModel saved as 'aqi_model.pkl'")
except Exception as e:
    print("Error saving model:", e)

# ============================================
# END
# ============================================
