import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set Global Random Seed for Reproducibility
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
file_path = 'data/germany_cleaned_load_data.csv'
data = pd.read_csv(file_path)

# Convert timestamps to datetime
data['utc_timestamp'] = pd.to_datetime(data['utc_timestamp'])
data.set_index('utc_timestamp', inplace=True)

# Select Target Variable
target_column = 'DE_load_actual_entsoe_transparency'
target_data = data[[target_column]]  # Keep as DataFrame

# Normalise Target Variable
scaler = MinMaxScaler(feature_range=(0, 1))
target_data_scaled = scaler.fit_transform(target_data)

print(f"Dataset Shape: {target_data.shape}")

# Split Data into Train (70%), Temp (30%) → Then Split Temp into Validation (15%) & Test (15%)
train_ratio = 0.7
train_data, temp_data = train_test_split(target_data_scaled, train_size=train_ratio, shuffle=False)
val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

print(f"Train Size: {len(train_data)}, Validation Size: {len(val_data)}, Test Size: {len(test_data)}")

# Function to Create Sequences for Time-Series Forecasting
def create_sequences(data, time_steps=24):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : i + time_steps])  # Past `time_steps` values
        y.append(data[i + time_steps])      # Target value at next step
    return np.array(X), np.array(y)

# Define Sequence Length (e.g., last 24 hours → predict next hour)
time_steps = 24

# Create Sequences for Train, Validation, and Test Sets
X_train, y_train = create_sequences(train_data, time_steps)
X_val, y_val = create_sequences(val_data, time_steps)
X_test, y_test_scaled = create_sequences(test_data, time_steps)  # Keep original scaled version

# Store a copy of the scaled test data to ensure we always use the correct scale
y_test_scaled_original = y_test_scaled.copy()

# Reshape Data for LSTM Input (Samples, Time Steps, Features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"Train Sequences: {X_train.shape}, Train Labels: {y_train.shape}")
print(f"Validation Sequences: {X_val.shape}, Validation Labels: {y_val.shape}")
print(f"Test Sequences: {X_test.shape}, Test Labels: {y_test_scaled.shape}")

# Function to evaluate models with proper unscaled metrics
def evaluate_model(model, X_test, y_test_scaled, scaler, model_name):
    # Make predictions on scaled data
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform to get actual values
    y_test_actual = scaler.inverse_transform(y_test_scaled.reshape(-1, 1))
    y_pred_actual = scaler.inverse_transform(y_pred_scaled)
    
    # Calculate metrics on unscaled values
    test_mape = mape(y_test_actual, y_pred_actual)
    test_mae = mean_absolute_error(y_test_actual, y_pred_actual)
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    test_r2 = r2_score(y_test_actual, y_pred_actual)
    
    # Print evaluation results
    print(f"\n{model_name} - Test Results:")
    print(f"MAPE: {test_mape:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    print(f"R² Score: {test_r2:.4f}")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual[:100], label="Actual Load", color="blue")
    plt.plot(y_pred_actual[:100], label="Predicted Load", color="orange", linestyle="dashed")
    plt.title(f"{model_name} - Forecast vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return test_mape, test_mae, test_rmse, test_r2

results = []

# Test 1: Simple LSTM Model (10 units)
def build_simple_lstm(input_shape):
    model = Sequential([
        LSTM(units=10, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Build and train simple model
input_shape = (X_train.shape[1], X_train.shape[2])
simple_lstm = build_simple_lstm(input_shape)

simple_lstm.fit(
    X_train, 
    y_train, 
    epochs=5,  # Keep at 5 epochs for simplicity as requested
    batch_size=32, 
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the simple model
print("\nTest 1: Simple LSTM (10 Units)")
simple_lstm_metrics = evaluate_model(
    simple_lstm, 
    X_test, 
    y_test_scaled_original, 
    scaler, 
    "Simple LSTM (10 Units)"
)
results.append(["Simple LSTM (10 Units)"] + list(simple_lstm_metrics))

# Test 2: LSTM with 50 Units
def build_lstm_with_50_units(input_shape):
    model = Sequential([
        LSTM(units=50, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Build and train model with 50 units
lstm_50 = build_lstm_with_50_units(input_shape)

lstm_50.fit(
    X_train, 
    y_train, 
    epochs=5,  # Keep at 5 epochs
    batch_size=32, 
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the model with 50 units
print("\nTest 2: LSTM (50 Units)")
lstm_50_metrics = evaluate_model(
    lstm_50, 
    X_test, 
    y_test_scaled_original, 
    scaler, 
    "LSTM (50 Units)"
)
results.append(["LSTM (50 Units)"] + list(lstm_50_metrics))

# Test 3: LSTM with 20% Dropout
def build_lstm_with_dropout_high(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Build and train model with 20% dropout
lstm_dropout_high = build_lstm_with_dropout_high(input_shape)

lstm_dropout_high.fit(
    X_train, 
    y_train, 
    epochs=5,  # Keep at 5 epochs
    batch_size=32, 
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the model with 20% dropout
print("\nTest 3: LSTM (50 Units + 20% Dropout)")
lstm_dropout_high_metrics = evaluate_model(
    lstm_dropout_high, 
    X_test, 
    y_test_scaled_original, 
    scaler, 
    "LSTM (50 Units + 20% Dropout)"
)
results.append(["LSTM (50 Units + 20% Dropout)"] + list(lstm_dropout_high_metrics))

# Test 4: LSTM with 10% Dropout
def build_lstm_with_dropout_low(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=False, input_shape=input_shape),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Build and train model with 10% dropout
lstm_dropout_low = build_lstm_with_dropout_low(input_shape)

lstm_dropout_low.fit(
    X_train, 
    y_train, 
    epochs=5,  # Keep at 5 epochs
    batch_size=32, 
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the model with 10% dropout
print("\nTest 4: LSTM (50 Units + 10% Dropout)")
lstm_dropout_low_metrics = evaluate_model(
    lstm_dropout_low, 
    X_test, 
    y_test_scaled_original, 
    scaler, 
    "LSTM (50 Units + 10% Dropout)"
)
results.append(["LSTM (50 Units + 10% Dropout)"] + list(lstm_dropout_low_metrics))

# Test 5: Stacked LSTM (50+30 units)
def build_stacked_lstm(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=30),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Build and train stacked LSTM model
stacked_lstm = build_stacked_lstm(input_shape)

stacked_lstm.fit(
    X_train, 
    y_train, 
    epochs=5,  # Keep at 5 epochs
    batch_size=32, 
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the stacked LSTM model
print("\nTest 5: Stacked LSTM (50+30 Units)")
stacked_lstm_metrics = evaluate_model(
    stacked_lstm, 
    X_test, 
    y_test_scaled_original, 
    scaler, 
    "Stacked LSTM (50+30 Units)"
)
results.append(["Stacked LSTM (50+30 Units)"] + list(stacked_lstm_metrics))

# Test 6: With Early Stopping (patience=3)
def build_lstm_for_es3(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Early stopping callback with patience=3
early_stopping_p3 = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# Build and train model with early stopping (patience=3)
lstm_es3 = build_lstm_for_es3(input_shape)

lstm_es3.fit(
    X_train, 
    y_train, 
    epochs=20,  # More epochs since we have early stopping
    batch_size=32, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping_p3],
    verbose=1
)

# Evaluate the model with early stopping (patience=3)
print("\nTest 6: LSTM with Early Stopping (patience=3)")
lstm_es3_metrics = evaluate_model(
    lstm_es3, 
    X_test, 
    y_test_scaled_original, 
    scaler, 
    "LSTM with Early Stopping (patience=3)"
)
results.append(["LSTM with Early Stopping (patience=3)"] + list(lstm_es3_metrics))

# Test 7: With Early Stopping (patience=5)
def build_lstm_for_es5(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Early stopping callback with patience=5
early_stopping_p5 = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Build and train model with early stopping (patience=5)
lstm_es5 = build_lstm_for_es5(input_shape)

lstm_es5.fit(
    X_train, 
    y_train, 
    epochs=20,  # More epochs since we have early stopping
    batch_size=32, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping_p5],
    verbose=1
)

# Evaluate the model with early stopping (patience=5)
print("\nTest 7: LSTM with Early Stopping (patience=5)")
lstm_es5_metrics = evaluate_model(
    lstm_es5, 
    X_test, 
    y_test_scaled_original, 
    scaler, 
    "LSTM with Early Stopping (patience=5)"
)
results.append(["LSTM with Early Stopping (patience=5)"] + list(lstm_es5_metrics))

# Create results DataFrame
results_df = pd.DataFrame(
    results, 
    columns=["Model", "MAPE", "MAE", "RMSE", "R²"]
)

# Sort by MAPE (lower is better)
results_df = results_df.sort_values("MAPE")

# Print comparison table
print("\nModel Comparison (sorted by MAPE):")
print(results_df.to_string(index=False))