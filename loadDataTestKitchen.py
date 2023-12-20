import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#df = pd.read_pickle('sensor_data/sensor_6223.pkl')
df = pd.read_pickle('kitchen_combined/sensor_combined.pkl')
#print(df)
df.to_csv("kitchen.csv",index=False)
# Function to classify sensors as 0 or 1 based on sign

def classify_sensor_column(column):
    return (column >= 0).astype(float)

# Iterate through sensor columns in the DataFrame and create a new DataFrame
binary_data = pd.DataFrame({'day_of_week':df['day_of_week'],'hour':df['hour']})
binary_data['value_5887']= df['value_5887'].apply(lambda x: 1 if x < 1.0 and x>0.0 else 0.0)
for col in df.columns[3:]:  # Skip the timestamp column
    binary_data[col] = classify_sensor_column(df[col])

# Display the resulting DataFrame
print(binary_data)

# Define the logical condition for cooking detection
cooking_condition = (binary_data[['value_5887','value_5893', 'value_6896', 'value_6635','value_6633','value_6632','value_6253']].any(axis=1))

# Create a new 1D array indicating cooking (1) or not cooking (0)
cooking_labels = cooking_condition.astype(int)

# Display the result
result_df = pd.DataFrame({'day_of_week': binary_data['day_of_week'],'hour': binary_data['hour'], 'cooking_label': cooking_labels, 'Stove Light':binary_data['value_5887'],'Kitchen Motion':binary_data['value_5893'], 'Microwave':binary_data['value_6896'],'Kettle':binary_data['value_6635'],'Sandwich Maker':binary_data['value_6633'],'Coffeemaker':binary_data['value_6632'],'Fridge':binary_data['value_6253']} )
print(result_df)
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