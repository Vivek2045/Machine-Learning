import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

# Ensure the model directory exists
os.makedirs('model', exist_ok=True)

# Load Dataset
data = pd.read_csv("house_dataset.csv")
print("Dataset Loaded Successfully")
print(data.head())

# Statistical Analysis
print("Statistical Analysis:")
print(data.describe())

# Check for null values or missing data
missing_data = data.isnull().sum()
print(f"Columns with missing values:\n{missing_data[missing_data > 0]}")

# Separate categorical and numerical columns
cat_columns = data.select_dtypes(include="object").columns
num_columns = data.select_dtypes(include="number").columns

print(f"Categorical columns: {list(cat_columns)}")
print(f"Numerical columns: {list(num_columns)}")

# Fill missing values
for col in num_columns:
    data[col].fillna(data[col].mean(), inplace=True)

for col in cat_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Confirm no missing values remain
print("Missing values after imputation:")
print(data.isnull().sum())

# Apply Label Encoding to all categorical columns
label_encoders = {}
for col in cat_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoders for future use

# Feature-Target Split
X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.2f}")

# Save Model
model_path = 'model/house_price_model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
