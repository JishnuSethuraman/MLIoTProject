import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import csv
import json
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

def process_data(file_path):
    # Read the csv file into a dataframe
    print("File found")
    tqdm.pandas()
    #df = pd.read_csv(file_path)
    df = pd.concat([chunk for chunk in tqdm(pd.read_csv(file_path, chunksize=1000), desc='Loading data')])
    print("csv file imported")
    # Convert the 'timestamp' column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print("Times stampped")
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
    df['part_of_day'] = pd.cut(df['timestamp'].dt.hour, 
                            bins=[0, 6, 12, 18, 24], 
                            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                            right=False)
    print("time features extracted")
    print(len(df))  # Length of the dataframe

    # Sort the dataframe by timestamp
    df_sorted = df.sort_values(by='timestamp')

    # Group the dataframe by sensor_id
    sensor_dfs = {sensor_id: group for sensor_id, group in df_sorted.groupby('sensor_id')}

    # Resample and average 'value' while keeping other columns as is
    for sensor_id, sensor_df in tqdm(sensor_dfs.items(), desc ="Sampling Data", unit="data"):
        sensor_df = sensor_df.resample('T', on='timestamp').agg({'value': 'mean', 'hour': 'first', 'day_of_week': 'first', 'part_of_day': 'first'})

        # Fill NaN values with 0
        sensor_df['value'].fillna(0, inplace=True)

        sensor_dfs[sensor_id] = sensor_df

    # Print the first 5 rows of each sensor's dataframe
    # for sensor_id, sensor_df in sensor_dfs.items():
    #     print(f"Sensor ID: {sensor_id}, DataFrame Size: {sensor_df.shape}")
    #     print(sensor_df.head(), "\n")
    
    return sensor_dfs

# Function to save data
def save_data(sensor_dfs, folder='sensor_data'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for sensor_id, sensor_df in sensor_dfs.items():
        with open(os.path.join(folder, f'sensor_{sensor_id}.pkl'), 'wb') as file:
            pickle.dump(sensor_df, file)

def align_sensors(sensor_dfs1, sensor_dfs2, sensor_ids):
    # Find the latest start date and earliest end date among the selected sensors
    latest_start = None
    earliest_end = None

    for sensor_id in sensor_ids:
        sensor_df1 = sensor_dfs1.get(sensor_id)
        sensor_df2 = sensor_dfs2.get(sensor_id)
        # Determine the start and end dates from both datasets for the sensor
        if sensor_df1 is not None:
            start_date1, end_date1 = sensor_df1.index.min(), sensor_df1.index.max()
            latest_start = max(latest_start, start_date1) if latest_start else start_date1
            earliest_end = min(earliest_end, end_date1) if earliest_end else end_date1
           

        if sensor_df2 is not None:
            start_date2, end_date2 = sensor_df2.index.min(), sensor_df2.index.max()
            latest_start = max(latest_start, start_date2) if latest_start else start_date2
            earliest_end = min(earliest_end, end_date2) if earliest_end else end_date2

    # Trim the dataframes in both datasets for the selected sensors
    aligned_sensors = {}
    for sensor_id in sensor_ids:
        if sensor_id in sensor_dfs1:
            aligned_sensors[sensor_id] = sensor_dfs1[sensor_id].loc[latest_start:earliest_end]
        if sensor_id in sensor_dfs2:
            aligned_sensors[sensor_id] = sensor_dfs2[sensor_id].loc[latest_start:earliest_end]

    return aligned_sensors

def normalize_and_aggregate(sensor_dfs, sensor_ids):
    normalized_data = []
    #print(sensor_dfs)
    timestamps = sensor_dfs[sensor_ids[0]]['hour']
    normalized_data.append(timestamps)
    for sensor_id in sensor_ids:
        # Select the 'value' column (already replaced NaNs with 0s)
        values = sensor_dfs[sensor_id]['value'].values.reshape(-1, 1)
        #print("word")
        # Normalize the values
        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(values)

        # Create a dataframe from the normalized values
        df_normalized = pd.DataFrame(normalized_values, columns=[f'value_{sensor_id}'], index=sensor_dfs[sensor_id].index)

        normalized_data.append(df_normalized)

    # Combine all normalized data into a single dataframe
    combined_df = pd.concat(normalized_data, axis=1)

    return combined_df


def main():
    #file_path = r'D:\vscodefiles\MLIoTProject\human_activity_sensor_data_in_home_environment\human_activity_raw_sensor_data\sensor_sample_int.csv'
    #file_path2 = r'D:\vscodefiles\MLIoTProject\human_activity_sensor_data_in_home_environment\human_activity_raw_sensor_data\sensor_sample_float.csv'
    file_path = "/Users/roshanpatel/Downloads/human_activity_raw_sensor_data/sensor_sample_int.csv"
    file_path2 = "/Users/roshanpatel/Downloads/human_activity_raw_sensor_data/sensor_sample_float.csv"
    print("Processing data...")
    """
    sensor_dfs = process_data(file_path2)
    sensorFloat = sensor_dfs
    # Convert DataFrame to a dictionary
    for key, value in sensorFloat.items():
        if isinstance(value, pd.DataFrame):
            sensorFloat[key] = value.to_dict(orient='records')
    # Save the dictionary to a JSON file
    with open('Sampled_float.json', 'w') as json_file:
        json.dump(sensorFloat, json_file)
    sensor_dfs2 = process_data(file_path)
    sensorInt = sensor_dfs2
    # Convert DataFrame to a dictionary
    for key, value in sensorInt.items():
        if isinstance(value, pd.DataFrame):
            sensorInt[key] = value.to_dict(orient='records')
    with open('Sampled_int.json', 'w') as json_file:
        json.dump(sensorInt, json_file)
    """
    # Load the dictionary from the JSON file
    with open('Sampled_float.json', 'r') as json_file:
        Floatdict = json.load(json_file)
        Floatdict = dict(Floatdict)
    # Convert DataFrame back to pandas DataFrame
    for key, value in Floatdict.items():
        if isinstance(value, list):  # Assuming list indicates a DataFrame
            Floatdict[key] = pd.DataFrame(value)
            # Load the dictionary from the JSON file
    with open('Sampled_int.json', 'r') as json_file:
        Intdict = json.load(json_file)
        Intdict = dict(Intdict)
    # Convert DataFrame back to pandas DataFrame
    for key, value in Intdict.items():
        if isinstance(value, list):  # Assuming list indicates a DataFrame
            Intdict[key] = pd.DataFrame(value)
    #selected_sensor_ids = [5896,5892,5895,7125, 5891,5889,6127,5887,6896,6635,6633,6632,6253]  # List of sensor IDs you want to align
    selected_sensor_ids = ['5893', '5887','6896','6635','6633','6632','6253']
    aligned_sensors = align_sensors(Intdict, Floatdict, selected_sensor_ids)
    save_data(aligned_sensors)
    normalized_combined_df = normalize_and_aggregate(aligned_sensors, selected_sensor_ids)
    save_data({'combined': normalized_combined_df}, folder='combined')

if __name__ == "__main__":
    main()

