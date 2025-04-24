import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

file_path = 'data/germany_cleaned_load_data.csv'  
germany_subset = pd.read_csv(file_path)

# Ensure the timestamp column is in datetime format
germany_subset['utc_timestamp'] = pd.to_datetime(germany_subset['utc_timestamp'])

# Set 'utc_timestamp' as the index 
germany_subset.set_index('utc_timestamp', inplace=True)

# Ensure you’re working with the correct column
target_variable = germany_subset['DE_load_actual_entsoe_transparency'].dropna()

# Plot ACF
plt.figure(figsize=(10, 6))
plot_acf(target_variable, lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.show()

# Plot PACF
plt.figure(figsize=(10, 6))
plot_pacf(target_variable, lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
