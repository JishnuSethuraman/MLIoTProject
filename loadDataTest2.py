import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df = pd.read_pickle('sensor_data/sensor_6223.pkl')
df = pd.read_pickle('combined/sensor_combined.pkl')
#print(df)
#df.to_csv("kitchen.csv",index=False)
# Function to classify sensors as 0 or 1 based on sign

def classify_sensor_column(column):
    return (column >= 0).astype(float)

# Iterate through sensor columns in the DataFrame and create a new DataFrame
binary_data = pd.DataFrame({'hour':df['hour']})
for col in df.columns[1:]:  # Skip the timestamp column
    binary_data[col] = classify_sensor_column(df[col])

# Display the resulting DataFrame
#print(binary_data)

# Define the logical condition for cooking detection
cooking_condition = (binary_data['value_5893'] == 1) & (binary_data[['value_5887', 'value_6896', 'value_6635','value_6633','value_6632','value_6253']].any(axis=1))

# Create a new 1D array indicating cooking (1) or not cooking (0)
cooking_labels = cooking_condition.astype(int)

# Display the result
result_df = pd.DataFrame({'hour': binary_data['hour'], 'cooking_label': cooking_labels})
#print(result_df)
result_df.to_pickle('kitchenData.pkl')

"""
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(result_df['index'], result_df['cooking_label'], marker='o', linestyle='-', color='b')
plt.title('Cooking Detection Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Cooking Label (1: Cooking, 0: Not Cooking)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


#plt.plot(df['value'])
"""