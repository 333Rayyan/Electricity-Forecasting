import pandas as pd

# Load the dataset
file_path = 'data/time_series_60min_singleindex.csv'  
data = pd.read_csv(file_path)

# Filter relevant columns for Germany
germany_columns = [
    'utc_timestamp',
    'DE_load_actual_entsoe_transparency',       # Target variable
    'DE_load_forecast_entsoe_transparency'      # Feature
]

germany_data = data[germany_columns]

# Convert utc_timestamp to datetime
germany_data['utc_timestamp'] = pd.to_datetime(germany_data['utc_timestamp'])

# **Remove all data from the year 2020**
germany_data = germany_data[germany_data['utc_timestamp'].dt.year != 2020]

# Drop rows where the target variable is missing
germany_data = germany_data.dropna(subset=['DE_load_actual_entsoe_transparency'])

# Fill missing values in the feature column with the mean
germany_data['DE_load_forecast_entsoe_transparency'].fillna(
    germany_data['DE_load_forecast_entsoe_transparency'].mean(), inplace=True
)

# Set datetime as the index
germany_data.set_index('utc_timestamp', inplace=True)

# Save the cleaned dataset to a new CSV file
output_path = 'data/germany_cleaned_load_data.csv'  
germany_data.to_csv(output_path, index=True)

print(f"Cleaned dataset saved successfully to {output_path}!")
