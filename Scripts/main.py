# Import functions
from Utils.helper import *
from Utils.train_test import demo_output

# Define model variables
image_size = (224, 224)
dropout_value = 0.5
dropout_value_2 = 0.2
num_outputs = 4
dense_lenght_sequence = 3

'''
# Create spatial model
spatial_model = create_spatial_model(image_size + (3,), num_classes=4, dropout_value=dropout_value)

# Create temporal model
temporal_model =  create_temporal_model(image_size + (3 * dense_lenght_sequence,), num_classes=4, dropout_value=dropout_value)

# Create CNN-RNN model
cnn_rnn_model = create_cnn_rnn_model(num_outputs)

# Create 3D-CNN model
cnn_3d_model = create_3d_cnn_model()

# Train spatial model
train_model(spatial_model, spatial_model_log, spatial_model_directory, type_of_model=1)

# Train temporal model
train_model(temporal_model, temporal_model_log, temporal_model_directory, type_of_model=2)

# Train CNN-RNN model
train_model(cnn_rnn_model, cnn_rnn_model_log, cnn_rnn_model_directory, type_of_model=3)

# Train 3D-CNN model
train_model(cnn_3d_model, cnn_3d_model_log, cnn_3d_model_directory, type_of_model = 4)

# Test two-stream convolutional model
test_two_stream_net(two_stream_conv_model_log, spatial_model_directory, temporal_model_directory, spatial_model_output, temporal_model_output, two_stream_conv_model_output)

# Test CNN-RNN model
#test_cnn(cnn_rnn_model_log, cnn_rnn_model_directory, cnn_rnn_model_output)

# Test CNN-RNN model
test_cnn(cnn_3d_model_log, cnn_3d_model_directory, cnn_3d_model_output, addition_axis=3, tensor_axis_split=3, type_of_model=4)

'''