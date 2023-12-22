import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential , load_model
from keras.layers import Dense, LSTM ,Dropout

data = pd.read_csv('data\\EURUSD_p3.csv')
# Extract OHLC data
features = data[['High', 'Low', 'Close']]
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


from tensorflow.keras.layers import Bidirectional

model = Sequential()
model.add(Bidirectional(LSTM(500, return_sequences=True), input_shape=(x_train.shape[1], 3)))
model.add(Bidirectional(LSTM(50, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(20))




# Compile the model
model.compile(optimizer='adam', loss='mse')
# Train the model
model.fit(x_train, y_train, batch_size=10, epochs=5) #  loss: 6.9001e-04
model.save('bid.h5')