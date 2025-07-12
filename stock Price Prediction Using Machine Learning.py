#!/usr/bin/env python
# coding: utf-8

# Step 1: Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 2: Download historical stock data
data = yf.download("AAPL", start="2015-01-01", end="2024-12-31", auto_adjust=False)
data = data[['Close']]
data.dropna(inplace=True)

# Step 3: Feature Engineering
data['Prev Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

# Step 4: Prepare input (X) and target (y)
X = data[['Prev Close']].rename(columns=lambda x:str(x))
y = data['Close']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 6: Train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Step 8: Plot actual vs predicted prices
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test.values, label='Actual Price', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted Price', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Stock Price - AAPL')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


