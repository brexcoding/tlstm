## tl.py ----> i.e ,   Tweaked_LSTM.py , i added a Bidirectional layer in the first neuron
###i want to implement the  moving averages in this model and also some patterns 
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential , load_model
from keras.layers import Dense, LSTM ,Dropout
from tensorflow.keras.layers import Bidirectional

data = pd.read_csv('data\\EURUSD_p.csv')
# Extract OHLC data
features = data[['Open', 'High', 'Low', 'volume', 'Close']]
features

# # Normalize OHLC data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(features)
# Create sequences
sequences = []
for i in range(20, len(normalized_data)):
    sequences.append(normalized_data[i - 20:i])

# Convert sequences to numpy array
sequences = np.array(sequences)

# Separate features and targets
x_train = sequences[:, :, :]  # Features ,the shape is 3d (number_of_samples, timesteps, features)
y_train = sequences[:, :, -1:]  # Targets
print(x_train.shape)
print(y_train.shape)



model = Sequential()
model.add(Bidirectional(LSTM(500, return_sequences=True), input_shape=(x_train.shape[1], 5)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True , activation="relu")) # Add another LSTM layer
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(20))




# Compile the model
model.compile(optimizer='adam', loss='mse')
# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5) #  loss: 6.9001e-04

model.save('tl.h5')