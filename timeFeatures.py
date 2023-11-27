import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

def process_data(file_path):
    # Read the csv file into a dataframe
    df = pd.read_csv(file_path)

    # Convert the 'timestamp' column to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
    df['part_of_day'] = pd.cut(df['timestamp'].dt.hour, 
                            bins=[0, 6, 12, 18, 24], 
                            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                            right=False)

    print(len(df))  # Length of the dataframe

    # Sort the dataframe by timestamp
    df_sorted = df.sort_values(by='timestamp')

    # Group the dataframe by sensor_id
    sensor_dfs = {sensor_id: group for sensor_id, group in df_sorted.groupby('sensor_id')}

    # Resample and average 'value' while keeping other columns as is
    for sensor_id, sensor_df in sensor_dfs.items():
        sensor_dfs[sensor_id] = sensor_df.resample('T', on='timestamp').agg({'value': 'mean', 'hour': 'first', 'day_of_week': 'first', 'part_of_day': 'first'})

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

def main():
    file_path = r'D:\vscodefiles\MLIoTProject\human_activity_sensor_data_in_home_environment\human_activity_raw_sensor_data\sensor_sample_float.csv'
    print("Processing data...")
    sensor_dfs = process_data(file_path)
    save_data(sensor_dfs)

if __name__ == "__main__":
    main()

# sensor_id_to_plot = '6634'  # Replace with an actual sensor_id

# if sensor_id_to_plot in sensor_dfs:
#     # Select one of the sensor dataframes to plot
#     sensor_df_to_plot = sensor_dfs[sensor_id_to_plot]

#     # Plotting
#     plt.figure(figsize=(10, 6))
#     plt.plot(sensor_df_to_plot.index, sensor_df_to_plot['value'], label=f'Sensor {sensor_id_to_plot}')
#     plt.xlabel('Time')
#     plt.ylabel('Value')
#     plt.title(f'Sensor Data over Time for Sensor ID {sensor_id_to_plot}')
#     plt.legend()
#     plt.show()
# else:
#     print(f"Sensor ID {sensor_id_to_plot} not found in the dataset.")

# # device = torch.device("cuda:0")

# class LSTMAttention(nn.Module):
#     def __init__(self, num_features, hidden_size, seq_len, output_size):
#         super(LSTMAttention, self).__init__()
        
#         self.num_features = num_features
#         self.hidden_size = hidden_size
#         self.seq_len = seq_len

#         self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True)
#         self.attention = nn.Linear(hidden_size, seq_len)
#         self.fc = nn.Linear(hidden_size * seq_len, output_size)

#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)  # Shape of lstm_out: [batch_size, seq_len, hidden_size]
#         attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
#         # Apply attention weights
#         attn_applied = torch.mul(lstm_out, attention_weights)

#         # Reshape and apply fully connected layer
#         fc_input = attn_applied.reshape(attn_applied.shape[0], -1)
#         output = self.fc(fc_input)

#         return output

# # Assuming you know the number of features and the sequence length
# num_features = len(df.columns)  # Number of features in your dataset
# hidden_size = 64  # Example size
# seq_len = 30  # Example sequence length
# output_size = 1  # Predicting a single value

# model = LSTMAttention(num_features, hidden_size, seq_len, output_size)
