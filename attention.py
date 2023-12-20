import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Permute, Dot, Flatten, multiply, Dropout
from keras import backend as K
from torch import dropout_

# Load data
org = pd.read_pickle("theSeries.pkl")
df = org[1:1440]

# Add Date column using .loc to avoid the SettingWithCopyWarning
df.loc[:, 'Date'] = range(1, len(df) + 1)

# Prepare data
close_data = df['combined'].values.reshape((-1, 1))

split_percent = 0.80
split = int(split_percent * len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
close_train_normalized = scaler.fit_transform(close_train)
close_test_normalized = scaler.transform(close_test)
#close_train_normalized = close_train
#close_test_normalized = close_test
# Function to create attention layer

def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(1, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

# Build LSTM model with attention
inputs = Input(shape=(1, 1))
lstm = LSTM(10, return_sequences=True)(inputs)

attention = attention_3d_block(lstm)

attention = Flatten()(attention)
attention = Dropout(0.3)(attention)
#output = Dense(1)(lstm)
output = Dense(1)(attention)

model = Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(close_train_normalized, close_train_normalized, epochs=50, batch_size=16, validation_split=0.1, shuffle=False)
import matplotlib.pyplot as plt
# Plot the validation loss
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Loss')
plt.title('Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Predictions
predictions = model.predict(close_test_normalized)

# Inverse transform to get original scale
# Reshape predictions to 2D array
predictions_reshaped = predictions.reshape(-1, 1)
predictions_original_scale = scaler.inverse_transform(predictions_reshaped)
#predictions_original_scale = predictions


# Plot training data
plt.figure(figsize=(12, 6))
plt.plot(date_train, close_train, label='Training Data', color='blue')

# Plot test data
plt.plot(date_test, close_test, label='Test Data', color='green')

# Plot predictions
plt.plot(date_test, predictions_original_scale, label='Predictions', color='red')

plt.title('LSTM with Attention: Training Data, Test Data, and Predictions')
plt.xlabel('Date')
plt.ylabel('Combined Value')
plt.legend()
plt.show()
