import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Load the cleaned dataset
file_path = 'data/germany_cleaned_load_data.csv'  
germany_data = pd.read_csv(file_path, index_col='utc_timestamp', parse_dates=True)

# Extract the target variable
target_variable = germany_data['DE_load_actual_entsoe_transparency']

# Plot the target variable to visualize trends or seasonality
plt.figure(figsize=(10, 6))
plt.plot(target_variable, label='Actual Load')
plt.title("Germany Actual Load Over Time")
plt.xlabel("Time")
plt.ylabel("Load (MW)")
plt.legend()
plt.show()

# Perform the Augmented Dickey-Fuller test
adf_result = adfuller(target_variable.dropna())  

# Print ADF test results
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value}")

# Interpretation of stationarity
if adf_result[1] < 0.05:
    print("\nThe data is stationary (p-value < 0.05).")
else:
    print("\nThe data is not stationary (p-value >= 0.05).")
    print("Consider differencing the data.")


    # Plot the differenced data
    plt.figure(figsize=(10, 6))
    plt.plot(germany_data['differenced'], label='Differenced Load')
    plt.title("Differenced Data")
    plt.xlabel("Time")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.show()

    # Run ADF test again on the differenced series
    adf_result_diff = adfuller(germany_data['differenced'].dropna())
    print("\nDifferenced Data ADF Test Results:")
    print("Differenced ADF Statistic:", adf_result_diff[0])
    print("Differenced p-value:", adf_result_diff[1])
    print("Critical Values:")
    for key, value in adf_result_diff[4].items():
        print(f"   {key}: {value}")

print("\nStationarity test completed.")
