import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from statistics import mode
df = pd.read_pickle('kitchenData.pkl')
daily_window = []
dfSize  = len(df)
windowLens =[]
#print(dfSize)
i=0
while(i<dfSize):
    j=i
    while(True):
        if(i==dfSize-1):
            window = df.iloc[j:i]
            daily_window.append(window)
            i+=1
            windowLens.append(len(window))
            break
        elif(df.at[i,'hour']>df.at[i+1,'hour']):
            window = df.iloc[j:i+1]
            daily_window.append(window)
            i+=1
            windowLens.append(len(window))
            break
        i+=1
lenMode = mode(windowLens)
daily_window[:] = [window for window in daily_window if len(window) == lenMode]
# Data Preparation and Training
for window in daily_window:
    X = window['hour'].values.reshape(-1, 1)
    y = window['cooking_label'].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Architecture
model = keras.Sequential([
    layers.LSTM(units=50, input_shape=(X_train_scaled.shape[1], 1)),
    layers.Dense(units=1, activation='sigmoid')
])

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluation
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {test_acc}')
