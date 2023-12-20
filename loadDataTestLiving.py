import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df = pd.read_pickle('sensor_data/sensor_6223.pkl')
df = pd.read_pickle('living_combined/sensor_combined.pkl')
#print(df)
df.to_csv("living.csv",index=False)
# Function to classify sensors as 0 or 1 based on sign

#def classify_sensor_column(column):
#    return (column >= 0).astype(float)

# Iterate through sensor columns in the DataFrame and create a new DataFrame
binary_data = pd.DataFrame({'day_of_week':df['day_of_week'],'hour':df['hour']})
binary_data['value_6127']= df['value_6127'].apply(lambda x: 1 if x < 0.2 else 0)
binary_data['value_5889']= df['value_5889'].apply(lambda x: 1 if x > 0.9 else 0)
#for col in df.columns[1:]:  # Skip the timestamp column
#    binary_data[col] = classify_sensor_column(df[col])

# Display the resulting DataFrame
#print(binary_data)

# Define the logical condition for cooking detection
living_condition = (binary_data[['value_6127', 'value_5889']].any(axis=1))

# Create a new 1D array indicating cooking (1) or not cooking (0)
living_labels = living_condition.astype(int)
living_labels = living_labels.replace(1,2)
# Display the result
result_df = pd.DataFrame({'day_of_week': binary_data['day_of_week'], 'hour':df['hour'],'living_label': living_labels, 'TV Light':binary_data['value_6127'],'Couch Pressure':binary_data['value_5889']})
print(result_df)
result_df.to_pickle('livingData.pkl')

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