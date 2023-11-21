import pandas as pd

file_path = r'D:\vscodefiles\MLIoTProject\human_activity_sensor_data_in_home_environment\human_activity_raw_sensor_data\sensor_sample_float.csv'

# Import the CSV file
df = pd.read_csv(file_path)

# Convert the 'timestamp' column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
df['part_of_day'] = pd.cut(df['timestamp'].dt.hour, 
                           bins=[0, 6, 12, 18, 24], 
                           labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                           right=False)

print(len(df))  # Display the first few rows to verify the changes

# Sort the dataframe by timestamp in ascending order
df_sorted = df.sort_values(by='timestamp')

# Group by sensor_id
sensor_dfs = {sensor_id: group for sensor_id, group in df_sorted.groupby('sensor_id')}

# Resample each sensor-specific dataframe by minute and average the values
for sensor_id, sensor_df in sensor_dfs.items():
    # Resample and average 'value' while keeping other columns as is
    sensor_dfs[sensor_id] = sensor_df.resample('T', on='timestamp').agg({'value': 'mean', 'hour': 'first', 'day_of_week': 'first', 'part_of_day': 'first'})

# Print the size of each dataframe and display the first few rows
for sensor_id, sensor_df in sensor_dfs.items():
    print(f"Sensor ID: {sensor_id}, DataFrame Size: {sensor_df.shape}")
    print(sensor_df.head(), "\n")