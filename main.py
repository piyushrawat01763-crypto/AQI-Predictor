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
