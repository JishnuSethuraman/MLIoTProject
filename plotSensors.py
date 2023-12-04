import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('rooms.csv')

# Assuming your columns are named Sensor1, Sensor2, ..., Sensor9
sensor_columns = ['value_5896', 'value_5892', 'value_7125','value_5895', 'value_5893', 'value_6253', 'value_5891', 'value_5889', 'value_6127']

# Plot each sensor on a separate plot
for sensor in sensor_columns:
    plt.figure()  # Create a new figure for each sensor
    plt.plot(df[sensor])
    plt.xlabel('Time Point')
    plt.ylabel(f'{sensor} Value')
    plt.title(f'Time Series for {sensor}')

# Plot a combined plot for the first three sensors
plt.figure()
for sensor in sensor_columns[:2]:
    plt.plot(df[sensor], label=sensor)
plt.xlabel('Time Point')
plt.ylabel('Sensor Value')
plt.title('Combined Time Series for Sensors 1, 2, and 3')
plt.legend()

plt.show()
