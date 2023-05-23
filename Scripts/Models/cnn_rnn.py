# Import libraries
from keras import layers, Input, Model

# Create CNN-RNN model
def create_cnn_rnn_model(num_classes, max_sequence_lenght = 5, num_features = 2048):

    # Model input
    frame_features_input = Input((max_sequence_lenght, num_features))
    mask_input = Input((max_sequence_lenght,), dtype="bool")

    # GRU Block
    gru1 = layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    gru2 = layers.GRU(8)(gru1)

    # Output Block
    dropout = layers.Dropout(0.4)(gru2)
    dense = layers.Dense(8, activation="relu")(dropout)
    output = layers.Dense(num_classes, activation="softmax")(dense)

    # Create and return model
    rnn_model = Model([frame_features_input, mask_input], output)
    return rnn_model