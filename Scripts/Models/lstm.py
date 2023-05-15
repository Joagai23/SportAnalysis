# Import libraries
from keras import Sequential
from keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, Dense, Lambda

# Create Long Recurrent Convolutional Network: Conv + LSTMmodel = Sequential()
def create_lstm_model(width, height, num_classes):
    input_shape = (width, height)
    model = Sequential()
    model.add(Lambda(lambda x: x[:,:,:,0], input_shape=(*input_shape, 1)))
    model.add(LSTM(units=256, return_sequences=True))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(LSTM(units=64))
    model.add(Dense(128))
    model.add(Dense(num_classes, activation='softmax'))

    return model