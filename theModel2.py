import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense, Layer, Flatten, Attention
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, LSTM, Dense, Attention, Concatenate
from keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from keras.layers import MaxPooling1D, Flatten
from keras.optimizers import Adam

def within_tolerance(y_true, y_pred, tolerance=0.05):
    """
    Custom metric to count the number of predictions within a tolerance of the true values.
    """
    diff = K.abs(y_true - y_pred)
    within_tolerance = K.cast(K.less_equal(diff, tolerance), dtype='float32')
    return K.mean(within_tolerance)


# Create a MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))


org = pd.read_pickle("theSeries.pkl")
df = org[1:1440]
close_data = df['combined'].values
close_data = close_data.reshape((-1,1))
close_data_normalized = scaler.fit_transform(close_data)

# Fit the scaler on the training data and transform both training and testing data
split_percent = 0.80
split = int(split_percent*len(close_data_normalized))

df['Date'] = range(1,len(df)+1)

#



close_train = close_data_normalized[:split]
close_test = close_data_normalized[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]

print(len(close_train))
print(len(close_test))

look_back = 10
"""
scaler = MinMaxScaler(feature_range=(0, 1))
close_train_normalized = scaler.fit_transform(close_train)
close_test_normalized = scaler.transform(close_test)
train_generator = TimeseriesGenerator(close_train_normalized, close_train_normalized, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test_normalized, close_test_normalized, length=look_back, batch_size=1)
"""

train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

print(len(train_generator))
print(len(test_generator))

print("labels generated")
"""
# Build the model with attention
model = Sequential()
model.add(LSTM(units=50, input_shape=(look_back, 1), return_sequences=True))

# Define an input layer for the attention mechanism
attention_input = Input(shape=(look_back, 50))

# Attention layer
attention_output = Attention(use_scale=True)([attention_input, model.layers[0].output])

# Concatenate the LSTM output and attention weights
merged = tf.keras.layers.Concatenate(axis=-1)([model.layers[0].output, attention_output])

model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(train_generator, epochs=50)

# Evaluate the model
loss = model.evaluate(test_generator)
print(f"Test Loss: {loss}")

def lstm_attention_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    lstm =  LSTM(100,activation='relu', return_sequences=True)(inputs)

    attention = Dense(1,activation="tanh")(lstm)
    attention = Flatten()(attention)
    attention = keras.layers.Activation('softmax')(attention)
    attention = keras.layers.RepeatVector(100)(attention)
    attention = keras.layers.Permute([2,1])(attention)
    attention = keras.layers.Multiply()([lstm,attention])
    attention = keras.layers.Lambda(lambda x: tf.reduce_sum(x,axis=1))(attention)

    outputs=Dense(1)(attention)

    model = keras.Model(inputs= inputs, outputs=outputs)
    model.compile(loss = 'mse',optimizer = 'adam')
    return model
attentionModel = lstm_attention_model(input_shape = (look_back,1))
attentionModel.fit(train_generator,epochs = 150,verbose=1)
"""

model = Sequential()
"""
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(100, 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
"""
model.add(
    LSTM(50,
        activation='relu',
        input_shape=(look_back,1))
)
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))
custom_optimizer = Adam(lr=0.001)
model.compile(custom_optimizer, loss='mse',metrics=[within_tolerance])
print("model compiled")
num_epochs = 30
history = model.fit(train_generator, epochs=num_epochs, verbose=1)
#plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['loss'], label='Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("model trained")
prediction = model.predict(test_generator)
#prediction = attentionModel.predict(test_generator)
print("model tested")
# Evaluate the model using the custom metric
evaluation = model.evaluate(test_generator)
print(f"Test Loss: {evaluation[0]}, Test Accuracy within ±0.05: {evaluation[1]}")
close_train = close_train.reshape((-1,1))
close_test = close_test.reshape((-1,1))
prediction = prediction.reshape((-1,1))
close_train = scaler.inverse_transform(close_train)
close_test = scaler.inverse_transform(close_test)
prediction = scaler.inverse_transform(prediction)
prediction = np.round(prediction)
#prediction = prediction
date_test = date_test[look_back:]
print(len(prediction))
# Assuming prediction is a one-dimensional array
predicted_label = np.argmax(prediction)

# Convert actual_labels to a one-dimensional array if it's not already
actual_labels = close_test[look_back:]
actual_labels = np.ravel(actual_labels)
"""
from sklearn.metrics import f1_score

# Calculate F1 Score
f1 = f1_score(actual_labels, np.repeat(predicted_label, len(actual_labels)), average='weighted')
print(f'F1 Score: {f1:.2f}')

"""
# Plot the training data in blue
plt.plot(date_train, close_train, label='Data', linestyle='-',color='blue')

# Plot the ground truth in green
plt.plot(date_test, close_test[look_back:], label='Ground Truth', linestyle='-', color='green')
# Plot the predicted data in orange
plt.plot(date_test, prediction, label='Prediction', linestyle='-', color='orange')


# Set plot labels and title
plt.xlabel('Time(minutes)')
plt.ylabel('Activity')
plt.title('LSTM')

# Add legend
plt.legend()
print("model plotting")
# Show the plot
plt.show()