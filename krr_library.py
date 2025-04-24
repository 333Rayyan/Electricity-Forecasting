import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# Load dataset
file_path = 'data/germany_cleaned_load_data.csv'
data = pd.read_csv(file_path)

# Convert timestamps to datetime
data['utc_timestamp'] = pd.to_datetime(data['utc_timestamp'])

# Extract time-based features
data['hour'] = data['utc_timestamp'].dt.hour  # Hour of the day (0-23)
data['day_of_week'] = data['utc_timestamp'].dt.dayofweek  # Day of the week (0=Monday, 6=Sunday)
data['month'] = data['utc_timestamp'].dt.month  # Month (1-12)

# Drop the original timestamp column
data = data.drop(columns=['utc_timestamp'])

# Features (X) and target variable (y)
X = data.drop(columns=['DE_load_actual_entsoe_transparency']).values  # Feature matrix
y = data['DE_load_actual_entsoe_transparency'].values  # Target variable

# Step 1: Split Data into Train (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

# Step 2: Tune Alpha & Gamma Using the Validation Set
best_alpha, best_gamma, best_mape = None, None, float("inf")

for alpha in [0.1, 1.0, 10.0]:  # Regularisation strength
    for gamma in [0.01, 0.1, 1.0]:  # Kernel spread
        krr = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
        krr.fit(X_train, y_train)  # Train on training set
        val_predictions = krr.predict(X_val)  # Predict on validation set
        val_mape = mape(y_val, val_predictions)  # Evaluate performance

        print(f"Alpha={alpha}, Gamma={gamma} → Validation MAPE: {val_mape:.4f}")

        if val_mape < best_mape:
            best_alpha, best_gamma, best_mape = alpha, gamma, val_mape  # Store best hyperparameters

print(f"\nBest Alpha: {best_alpha}, Best Gamma: {best_gamma}, Best Validation MAPE: {best_mape:.4f}")

# Step 3: Train Final Model on Best Alpha & Gamma
final_krr = KernelRidge(kernel='rbf', alpha=best_alpha, gamma=best_gamma)
final_krr.fit(X_train, y_train)

# Step 4: Test Model on Unseen Test Data
test_predictions = final_krr.predict(X_test)
test_mape = mape(y_test, test_predictions)

print(f"\nFinal Test MAPE: {test_mape:.4f}")

# Step 5: Plot Predictions vs Actual for First 100 Points
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], label="Actual Load", color="blue")
plt.plot(test_predictions[:100], label="Predicted Load", color="orange", linestyle="dashed")
plt.title("Kernel Ridge Regression Forecast vs Actual (First 100 Points)")
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.legend()
plt.grid(True)
plt.show()
