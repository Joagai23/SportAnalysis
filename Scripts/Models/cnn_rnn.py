# Import libraries
from keras import layers, Input, Model

# Create CNN-RNN model
def create_cnn_rnn_model(num_classes, max_sequence_lenght = 5, num_features = 2048):

    # Model input
    frame_features_input = Input((max_sequence_lenght, num_features))
    mask_input = Input((max_sequence_lenght,), dtype="bool")

    # GRU Block
    x = layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = layers.GRU(8)(x)

    # Output Block
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(8, activation="relu")(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    # Create and return model
    rnn_model = Model([frame_features_input, mask_input], output)
    return rnn_model