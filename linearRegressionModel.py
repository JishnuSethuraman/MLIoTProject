import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import pandas as pd

# Assuming close_train is your training data
# Assuming close_test is your test data
# Assuming look_back is the length of your input sequences
org = pd.read_pickle("theSeries.pkl")
df = org[1:1440]

df['Date'] = range(1,len(df)+1)
close_data = df['combined'].values
close_data = close_data.reshape((-1,1))

split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

print(len(close_train))
print(len(close_test))

look_back = 3
# Prepare training data
X_train = np.array([close_train[i:i+look_back] for i in range(len(close_train)-look_back)])
y_train = close_train[look_back:]

# Prepare test data
X_test = np.array([close_test[i:i+look_back] for i in range(len(close_test)-look_back)])
y_test = close_test[look_back:]

# Reshape the data to fit the Conv1D layer
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, look_back)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, look_back)).reshape(X_test.shape)

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))  # Linear activation for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)

# Evaluate the model on the test data
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {loss}')

# Make predictions on the test data
predictions = model.predict(X_test)

# Plot the results
plt.plot(date_test[look_back:], y_test, label='Actual', linestyle='-', marker='o', color='blue')
plt.plot(date_test[look_back:], predictions, label='CNN Prediction', linestyle='-', marker='o', color='orange')

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('CNN Prediction vs. Actual Close Prices')
plt.legend()
plt.show()
