# Import libraries
from keras import layers, Input, Model

# Create 3D-CNN model
def create_3d_cnn_model(width = 224, height = 224, depth = 15, num_classes = 4):

    # Model input
    inputs = Input((width, height, depth, 1))

    # 3D Block 1
    conv1 = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    pool1 = layers.MaxPool3D(pool_size=2)(conv1)
    norm1 = layers.BatchNormalization()(pool1)

    # 3D Block 2
    conv2 = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(norm1)
    pool2 = layers.MaxPool3D(pool_size=2)(conv2)
    norm2 = layers.BatchNormalization()(pool2)

    # Output Block
    pool5 = layers.GlobalAveragePooling3D()(norm2)
    dense = layers.Dense(units=128, activation="relu")(pool5)
    dropout = layers.Dropout(0.3)(dense)
    output = layers.Dense(num_classes, activation="sigmoid")(dropout)

    # Create and return the model
    return Model(inputs, output)