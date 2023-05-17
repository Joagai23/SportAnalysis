# Import functions
from Models.spatial_stream_conv import create_spatial_model
from Models.temporal_stream_conv import create_temporal_model
from Models.cnn_rnn import create_cnn_rnn_model
from Utils.train_test import train_model, test_two_stream_net, test_sequence
from Utils.helper import *
from keras.utils import plot_model

# Define model variables
image_size = (224, 224)
dropout_value = 0.5
dropout_value_2 = 0.2
num_outputs = 4
dense_lenght_sequence = 3
lcrn_lenght_sequence = 5

# Create CNN-RNN model
cnn_rnn_model = create_cnn_rnn_model(num_outputs)

# Train CNN-RNN model
train_model(cnn_rnn_model, cnn_rnn_model_log, cnn_rnn_model_directory, type_of_model=3)

'''
# Create spatial model
spatial_model = create_spatial_model(image_size + (3,), num_classes=4, dropout_value=dropout_value)

# Create temporal model
temporal_model =  create_temporal_model(image_size + (3 * dense_lenght_sequence,), num_classes=4, dropout_value=dropout_value)

# Train spatial model
train_model(spatial_model, spatial_model_log, spatial_model_directory, type_of_model=1)

# Train temporal model
train_model(temporal_model, temporal_model_log, temporal_model_directory, type_of_model=2)

# Train LCRN model
train_model(lcrn_model, lcrn_model_log, lcrn_model_directory, type_of_model=3)

# Test two-stream convolutional model
test_two_stream_net(two_stream_conv_model_log, spatial_model_directory, temporal_model_directory, spatial_model_output, temporal_model_output, two_stream_conv_model_output)
'''