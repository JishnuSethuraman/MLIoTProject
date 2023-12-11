import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
org = pd.read_pickle("theSeries.pkl")
df = org[1:10080]
df['Date'] = range(1,len(df)+1)
close_data = df['combined'].values
close_data = close_data.reshape((-1,1))

split_percent = 0.70
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

print(len(close_train))
print(len(close_test))

look_back = 3

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

print(len(train_generator))
print(len(test_generator))
print("labels generated")
from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(
    LSTM(10,
        activation='relu',
        input_shape=(look_back,1))
)
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print("model compiled")
num_epochs = 25
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
print("model trained")
prediction = model.predict(test_generator)
print("model tested")
close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))
prediction = prediction
date_test = date_test[look_back:]
print(len(prediction))
# Plot the training data in blue
plt.plot(date_train, close_train, label='Data', linestyle='-', marker = 'o',color='blue')
# Plot the predicted data in orange
plt.plot(date_test, prediction, label='Prediction', linestyle='-', marker = 'o', color='orange')
# Plot the ground truth in green
plt.plot(date_test, close_test[look_back:], label='Ground Truth', linestyle='-', marker = 'o',color='green')


# Set plot labels and title
plt.xlabel('Time(minutes)')
plt.ylabel('Activity')
plt.title('Activity Prediction')

# Add legend
plt.legend()
print("model plotting")
# Show the plot
plt.show()