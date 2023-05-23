# Import functions
from Models.cnn_3d import create_3d_cnn_model
from Utils.train_test import train_model, test_cnn_rnn
from Utils.helper import *
from keras.utils import plot_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]) # Notice here
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


# Define model variables
image_size = (224, 224)
dropout_value = 0.5
dropout_value_2 = 0.2
num_outputs = 4
dense_lenght_sequence = 3

# Create 3D-CNN model
cnn_3d_model = create_3d_cnn_model()
plot_model(cnn_3d_model, show_shapes=True)

# Train 3D-CNN model
train_model(cnn_3d_model, cnn_3d_model_log, cnn_3d_model_directory, type_of_model = 4)

'''
# Create spatial model
spatial_model = create_spatial_model(image_size + (3,), num_classes=4, dropout_value=dropout_value)

# Create temporal model
temporal_model =  create_temporal_model(image_size + (3 * dense_lenght_sequence,), num_classes=4, dropout_value=dropout_value)

# Create CNN-RNN model
cnn_rnn_model = create_cnn_rnn_model(num_outputs)

# Train spatial model
train_model(spatial_model, spatial_model_log, spatial_model_directory, type_of_model=1)

# Train temporal model
train_model(temporal_model, temporal_model_log, temporal_model_directory, type_of_model=2)

# Train CNN-RNN model
train_model(cnn_rnn_model, cnn_rnn_model_log, cnn_rnn_model_directory, type_of_model=3)

# Test two-stream convolutional model
test_two_stream_net(two_stream_conv_model_log, spatial_model_directory, temporal_model_directory, spatial_model_output, temporal_model_output, two_stream_conv_model_output)

# Test CNN-RNN model
#test_cnn_rnn(cnn_rnn_model_log, cnn_rnn_model_directory, cnn_rnn_model_output)

'''