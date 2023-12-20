from keras.layers import Layer
from keras import backend as K
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
class Attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)
model = Sequential()
model.add(Embedding(max_words, emb_dim, input_length=max_len))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Attention(return_sequences=True)) # receive 3D and output 3D
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile('adam', 'binary_crossentropy')
model.fit(X,Y, epochs=3)