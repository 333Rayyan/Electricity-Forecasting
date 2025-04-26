import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error as mape
import time

# Load dataset
file_path = 'data/germany_cleaned_load_data.csv'
data = pd.read_csv(file_path)

# Convert timestamp and set as index
data['utc_timestamp'] = pd.to_datetime(data['utc_timestamp'])
data.set_index('utc_timestamp', inplace=True)

# Target variable
target_variable = data['DE_load_actual_entsoe_transparency']

# Train-Test Split
train_size = int(len(target_variable) * 0.8)  # 80% training, 20% testing
train, test = target_variable[:train_size], target_variable[train_size:]

# Step 1: Optimized ARMA Parameter Selection (FAST)
def find_best_arma(train, max_p=4, max_q=4, timeout=10):
    best_aic = float("inf")
    best_order = None
    for p in range(1, max_p+1):
        for q in range(1, max_q+1):
            try:
                start_time = time.time()
                model = ARIMA(train, order=(p, 0, q))
                arma_model = model.fit()
                elapsed_time = time.time() - start_time
                
                # Skip models that take too long
                if elapsed_time > timeout:
                    print(f"Skipping (p={p}, q={q}) due to timeout ({elapsed_time:.2f}s)")
                    continue
                
                if arma_model.aic < best_aic:
                    best_aic = arma_model.aic
                    best_order = (p, q)
                    print(f"New Best Model: p={p}, q={q}, AIC={best_aic:.2f}")
            except:
                continue
    return best_order

p, q = find_best_arma(train)
print(f"Final ARMA Order: p={p}, q={q}")

# Step 2: Train ARMA Model (NO ROLLING FORECAST)
print("Training ARMA model...")
arma_model = ARIMA(train, order=(p, 0, q))
arma_fit = arma_model.fit()

# Step 3: Multi-Step Forecast (FAST)
print("Generating Forecasts...")
forecast = arma_fit.forecast(steps=len(test))

# Convert predictions to Series
forecast_series = pd.Series(forecast, index=test.index)

# Step 4: Evaluate Model Performance
error = mape(test, forecast_series)
print(f'MAPE: {error:.2f}')

# Step 5: Plot Full Forecast vs Actual
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label="Actual Load", color="blue")
plt.plot(test.index, forecast_series, label="Predicted Load", color="orange", linestyle="dashed")
plt.title("Optimized Faster ARMA Model Forecast vs Actual")
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Zoomed-in 24-Hour Prediction
start_zoom = test.index[0]
end_zoom = start_zoom + pd.Timedelta(hours=24)

plt.figure(figsize=(10, 5))
plt.plot(test.loc[start_zoom:end_zoom], label="Actual Load", color="blue")
plt.plot(forecast_series.loc[start_zoom:end_zoom], label="Predicted Load", color="orange", linestyle="dashed")
plt.title(f"Zoomed-in: 24-Hour ARMA Forecast ({start_zoom.date()})")
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.legend()
plt.grid(True)
plt.show()
