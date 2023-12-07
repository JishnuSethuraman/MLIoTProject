import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('kitchen.csv')

# Assuming your columns are named Sensor1, Sensor2, ..., Sensor9
sensor_columns = ['value_5893', 'value_5887','value_6896','value_6635','value_6633','value_6632','value_6253']

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
