# Import libraries
from keras import Sequential
from keras.layers import TimeDistributed, Convolution2D, MaxPooling2D, Flatten, Dropout, LSTM, Dense

# Create Long Recurrent Convolutional Network: Conv + LSTM
def create_lcrn_model(width, height, channels, num_classes, dropout_value_1, dropout_value_2):
    
    # Initialize model
    model = Sequential()
  
    # Block 1
    model.add(TimeDistributed(Convolution2D(32, (7,7), strides=(2, 2), padding='same', activation='relu'), input_shape=(1, width, height, channels)))
    model.add(TimeDistributed(Convolution2D(32, (3,3), kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # Block 2
    model.add(TimeDistributed(Convolution2D(64, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Convolution2D(64, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # Block 3
    model.add(TimeDistributed(Convolution2D(128, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Convolution2D(128, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

    # Block 4
    model.add(TimeDistributed(Convolution2D(256, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Convolution2D(256, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Flatten()))

    # LSTM Block
    model.add(Dropout(dropout_value_1))
    model.add(LSTM(512, return_sequences=False, dropout=dropout_value_2))

    # Output Block
    model.add(Dense(num_classes, activation='softmax'))        
    return model