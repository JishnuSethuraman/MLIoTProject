import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_pickle('sensor_data/sensor_6223.pkl')
df = pd.read_pickle('combined/sensor_combined.pkl')
df.to_csv("rooms.csv",index=False)

#plt.plot(df['value'])