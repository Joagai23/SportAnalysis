# Import libraries
from keras import layers, Input, Model

# Create convolutional base
def create_spatial_model(input_shape, num_classes, dropout_value):

    # Model input
    inputs = Input(shape=input_shape)
    
    # Conv 1 Block
    conv1 = layers.Conv2D(96, (7, 7), strides=2, activation='relu')(inputs)
    norm1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPool2D((3, 3), strides=(2,2))(norm1)

    # Conv 2 Block
    conv2 = layers.Conv2D(256, (5, 5), strides=2, activation='relu')(pool1)
    norm2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPool2D((3, 3), strides=(2, 2))(norm2)

    # Conv 3 Block
    conv3 = layers.Conv2D(512, (3, 3), strides=1, activation='relu')(pool2)

    # Conv 4 Block
    conv4 = layers.Conv2D(512, (3, 3), strides=1, activation='relu')(conv3)

    # Conv 5 Block
    conv5 = layers.Conv2D(512, (3, 3), strides=1, activation='relu')(conv4)
    pool5 = layers.MaxPool2D((3, 3), strides=(2, 2))(conv5)
    flatten = layers.Flatten()(pool5)

    # Full 6 Block
    dense6 = layers.Dense(4096, activation='relu')(flatten)
    dropout6 = layers.Dropout(dropout_value)(dense6)

    # Full 7 Block
    dense7 = layers.Dense(2048, activation='relu')(dropout6)
    dropout7 = layers.Dropout(dropout_value)(dense7)

    # Output Block
    outputs = layers.Dense(num_classes, activation='softmax')(dropout7)
    return Model(inputs, outputs)