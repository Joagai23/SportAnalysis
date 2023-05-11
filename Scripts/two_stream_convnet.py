# Import libraries
import tensorflow as tf
import time
from keras import optimizers, losses, metrics, models
from datetime import datetime

# Import functions
from spatial_stream_conv import create_spatial_model
from temporal_stream_conv import create_temporal_model
from directory_image_manager import get_training_data, get_test_frames_by_dense, temporal_model_log, spatial_model_log, temporal_model_directory, spatial_model_directory, two_stream_conv_model_log
from log_writer import write_log

# Define model variables
image_size = (224, 224)
dropout_value = 0.3
num_outputs = 4
epochs = [3000, 2000, 1000]
learning_rates = [1e-3, 1e-4, 1e-5]
lenght_sequence = 3

# Create spatial model
spatial_model = create_spatial_model(image_size + (3,), num_classes=4, dropout_value=dropout_value)

# Create temporal model
temporal_model =  create_temporal_model(image_size + (3 * lenght_sequence,), num_classes=4, dropout_value=dropout_value)

# Train model function. Inputs = model, log_file, type_of_model (1 = spatial, 2 = temporal)
def train_model(model, log_file, model_directory, type_of_model = 1):

    # Instantiate loss function
    loss_function = losses.CategoricalCrossentropy()

    # Instantiate metrics
    training_metrics = metrics.CategoricalAccuracy()

    # Zip epoch and learning rate pair and iterate
    for epoch, learn_rate in zip(epochs, learning_rates):
        
        # Instantiate optimizer
        optimizer = optimizers.SGD(learning_rate=learn_rate)

        # Print Iteration Phase
        write_log(
            "Start of %d iterations with learning rate %.4f"
            % (epoch, learn_rate), log_file
        )

        # Get epoch starting time
        start_time = time.time()

        # Iterate batches
        for step in range(epoch):

            # Obtain training inputs and outputs
            x_batch_train, y_batch_train = get_training_data(type_of_model)

            # Feed inputs to model
            for x, y in zip(x_batch_train, y_batch_train):
                
                # Record operations with Gradient Tape
                with tf.GradientTape() as tape:
                    
                    # Forward pass
                    prediction = model(x, training = True)

                    # Obtain tensor value from shape=(1, 4) to shape(4,)
                    prediction = prediction[0]

                    # Compute loss value
                    loss_value = loss_function(y, prediction)

                # Retrieve gradients of trainable variables
                gradients = tape.gradient(loss_value, model.trainable_weights)

                # Run gradient descent step
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))

                # Update training metric
                training_metrics.update_state(y, prediction)

            # Log every 100 iterations
            if step % 100 == 0 and step != 0:
                write_log(str(datetime.now()), log_file)
                write_log(
                    "Training loss at step %d: %f"
                    % (step, float(loss_value.numpy())), log_file
                )
                write_log(
                    "Training categorical accuracy at step %d: %f"
                    % (step, float(training_metrics.result())), log_file
                )

        # Save model
        model.save(model_directory + "_%d_%d" % (epoch, step))       

        # Reset training metrics at the end of each epoch
        training_metrics.reset_states()

        # Show training time for every epoch
        write_log("Time taken: %.2fs" % (time.time() - start_time), log_file)

# Test model
def test_two_stream_net(log_file, spatial_model_directory, temporal_model_directory):

    # Set number of testing iterations
    num_iterations = 10

    # Set lenght of frames per sequence to test
    len_sequence = 15

    # Load models
    #spatial_model = models.load_model(spatial_model_directory)
    #temporal_model = models.load_model(temporal_model_directory)

    # Log starting time of testing process
    write_log(str("Start of Two-Stream Convolutional Network training at ", str(datetime.now())), log_file)

    # Iterate testing process
    for iteration in range(num_iterations):

        # Print Iteration Phase
        write_log(
            "Start of iteration %d of testing"
            % (iteration), log_file
        )

        # Obtain testing batches for current iteration
        # Batches are the lenght of the number of lines in testing file = 47
        spatial_x_batch_test, temporal_x_batch_test, y_batch_test = get_test_frames_by_dense(len_sequence)

        # Check the lenght of batches is the same
        # If not break iteration
        len_batch = 0
        if(len(spatial_x_batch_test) == len(temporal_x_batch_test) == len(y_batch_test)):
            len_batch = len(spatial_x_batch_test)
        else:
            write_log("Batch lenght error! at", str(datetime.now()), log_file)
            write_log("Spatial batch is %d, temporal batch is %d, label batch is %d" % (len(spatial_x_batch_test), len(temporal_x_batch_test), len(y_batch_test)), log_file)
            return
        
        print("Lenght of batches = ", len_batch)
        
        for mini_batch in range(len_batch):

            # Set mini-batch list values (len = len_sequence = 15)
            spatial_x_mini_batch_test = spatial_x_batch_test[mini_batch]
            spatial_x_mini_batch_test = spatial_x_batch_test[mini_batch]
            y_mini_batch_test = y_batch_test[mini_batch]

            # Check the lenght of batches is the same
            # If not break iteration
            len_mini_batch = 0
            if(len(spatial_x_mini_batch_test) == len(temporal_x_batch_test) == len(y_batch_test)):
                len_mini_batch = len(spatial_x_mini_batch_test)
            else:
                write_log("Mini-Batch lenght error! at", str(datetime.now()), log_file)
                write_log("Spatial mini-batch is %d, temporal mini-batch is %d, label mini-batch is %d" % (len(spatial_x_mini_batch_test), len(spatial_x_mini_batch_test), len(y_mini_batch_test)), log_file)
                return
            
            print("Lenght of mini-batches = ", len_mini_batch)

# Train spatial model
#train_model(spatial_model, spatial_model_log, spatial_model_directory, type_of_model=1)

# Train temporal model
#train_model(temporal_model, temporal_model_log, temporal_model_directory, type_of_model=2)

# Test two-stream convolutional model
test_two_stream_net(two_stream_conv_model_log, spatial_model_directory, temporal_model_directory)