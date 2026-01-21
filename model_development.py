"""
model_development.py
-------------------
House Price Prediction System - Model Development

This script loads the dataset, preprocesses the data, trains a regression model, evaluates it, and saves the trained model to disk.

Author: Nweze Ayinachiso
Date: 2026-01-21
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib


# 1. Load the dataset
data = pd.read_csv('train.csv')

# 2. Feature selection (choose 6 from the recommended 9)
# Features: OverallQual, GrLivArea, TotalBsmtSF, GarageCars, BedroomAbvGr, Neighborhood, SalePrice
selected_features = [
	'OverallQual',      # Overall material and finish quality (numeric)
	'GrLivArea',        # Above grade (ground) living area square feet (numeric)
	'TotalBsmtSF',     # Total square feet of basement area (numeric)
	'GarageCars',      # Size of garage in car capacity (numeric)
	'BedroomAbvGr',    # Number of bedrooms above basement level (numeric)
	'Neighborhood',    # Physical locations within Ames city limits (categorical)
]
target = 'SalePrice'

# 2a. Subset the dataframe
df = data[selected_features + [target]].copy()

# 2b. Handle missing values
# For numeric: fill with median; for categorical: fill with mode
numeric_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'BedroomAbvGr']
categorical_features = ['Neighborhood']

numeric_transformer = Pipeline([
	('imputer', SimpleImputer(strategy='median')),
	('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
	('imputer', SimpleImputer(strategy='most_frequent')),
	('encoder', OneHotEncoder(handle_unknown='ignore'))
])


# Combine transformers for preprocessing
preprocessor = ColumnTransformer([
	('num', numeric_transformer, numeric_features),
	('cat', categorical_transformer, categorical_features)
])

# 3. Split data into train and test sets
X = df[selected_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create a pipeline with preprocessing and regression model
model = Pipeline([
	('preprocessor', preprocessor),
	('regressor', LinearRegression())
])

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict and evaluate
y_pred = model.predict(X_test)

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")


# 7. Save the trained model to disk in the /model directory as .pkl for submission
import os
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/house_price_model.pkl')
print("Trained model saved as 'model/house_price_model.pkl'.")


# 8. Demonstrate reloading the model (no retraining required)
# (Best practice: reload in a separate script, but shown here for completeness)
loaded_model = joblib.load('model/house_price_model.pkl')
sample_pred = loaded_model.predict(X_test[:5])
print("Sample predictions from reloaded model:", sample_pred)
