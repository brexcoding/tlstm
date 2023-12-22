## retrain tweaked lstmn 

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential , load_model
from keras.layers import Dense, LSTM

data = pd.read_csv('data\\EURUSD_p8.csv')
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

model = load_model('tl.h5')
model.fit(x_train, y_train, batch_size=1, epochs=5)
model.save('tl1.h5')
