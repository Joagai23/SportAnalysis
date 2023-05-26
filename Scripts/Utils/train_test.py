# Import libraries
import tensorflow as tf
import numpy as np
import time
from keras import optimizers, losses, metrics, models, applications, Input, Model
from keras.applications.inception_v3 import preprocess_input
from datetime import datetime
from .helper import get_training_data, get_test_frames_by_dense, get_mean_output, get_frames_sequence, create_batch_mask, resize_volume
from .log_writer import write_log

# Define training variables
epochs = [3000, 2000, 1000]
learning_rates = [1e-3, 1e-4, 1e-5]

# Feature Extractor
def build_feature_extractor(image_size = 224):
    feature_extractor = applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(image_size, image_size, 3),
    )

    inputs = Input((image_size, image_size, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return Model(inputs, outputs, name="feature_extractor")

# Train model function. Inputs = model, log_file, type_of_model (1 = spatial, 2 = temporal, 3 = cnn-rnn, 4 = 3d-cnn)
def train_model(model, log_file, model_directory, type_of_model = 1, num_features = 2048, len_sequence = 5):

    # Instantiate loss function
    loss_function = losses.CategoricalCrossentropy()

    # Instantiate metrics
    training_metrics = metrics.CategoricalAccuracy()

    # Initialize CNN-RNN function
    if(type_of_model == 3):
        feature_extractor = build_feature_extractor()

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
            x_batch_train, y_batch_train = get_training_data(type_of_model, len_sequence)

            # Feed inputs to model
            for x, y in zip(x_batch_train, y_batch_train):

                # Record operations with Gradient Tape
                with tf.GradientTape() as tape:
                    
                    # Modify input if CNN-RNN
                    if type_of_model == 3:
                        x = feature_extractor(x)
                        x = x[None, :]
                        mask = create_batch_mask(len_sequence)
                        prediction = model((x, mask), training = True)
                    else:
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

            print(
                "Training loss at step %d: %f"
                % (step, float(loss_value.numpy()))
            )
            print(
                "Training categorical accuracy at step %d: %f"
                % (step, float(training_metrics.result()))
            )

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

# Test Two-Stream Convolutional Network model
def test_two_stream_net(log_file, spatial_model_directory, temporal_model_directory, spatial_output, temporal_output, two_stream_output):

    # Set number of testing iterations
    num_iterations = 500

    # Set lenght of frames per sequence to test
    len_sequence = 15

    # Load models
    spatial_model = models.load_model(spatial_model_directory, compile=False)
    temporal_model = models.load_model(temporal_model_directory, compile=False)

    # Log starting time of testing process
    write_log(message = "Start of Two-Stream Convolutional Network testing at %s" % (str(datetime.now())), file_name = log_file)

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
        
        # Define Categorical Accuracy Metrics
        conv_net_cat_acc = metrics.CategoricalAccuracy()
        spatial_net_cat_acc = metrics.CategoricalAccuracy()
        temporal_net_cat_acc = metrics.CategoricalAccuracy()
        
        for mini_batch in range(len_batch):

            # Set mini-batch list values (len = len_sequence = 15)
            spatial_x_mini_batch_test = spatial_x_batch_test[mini_batch]
            temporal_x_mini_batch_test = temporal_x_batch_test[mini_batch]
            y_mini_batch_test = y_batch_test[mini_batch]

            # Initialize model output list
            spatial_output_list = []
            temporal_output_list = []

            # Check the lenght of batches is the same
            # If not break iteration
            len_mini_batch = 0
            if(len(spatial_x_mini_batch_test) == len(temporal_x_mini_batch_test)):
                len_mini_batch = len(spatial_x_mini_batch_test)
            else:
                write_log(message = "Mini-Batch lenght error! at %s" % (str(datetime.now())), file_name = log_file)
                write_log("Spatial mini-batch is %d, temporal mini-batch is %d, label mini-batch is %d" % (len(spatial_x_mini_batch_test), len(temporal_x_mini_batch_test), len(y_mini_batch_test)), log_file)
                return
            
            # Iterate inputs and feed them to the models
            for input in range(len_mini_batch):

                # Append output values to mini_batch list
                spatial_output_list.append(spatial_model(spatial_x_mini_batch_test[input], training = False)[0])
                temporal_output_list.append(temporal_model(temporal_x_mini_batch_test[input], training = False)[0])

            # Average output list values
            spatial_mean_prediction = get_mean_output(spatial_output_list)
            temporal_mean_prediction = get_mean_output(temporal_output_list)

            # Fuse output predictions (Regular average)
            two_stream_prediction = get_mean_output([spatial_mean_prediction, temporal_mean_prediction])

            # Calculate metrics
            conv_net_cat_acc.update_state(two_stream_prediction, y_mini_batch_test)
            spatial_net_cat_acc.update_state(spatial_mean_prediction, y_mini_batch_test)
            temporal_net_cat_acc.update_state(temporal_mean_prediction, y_mini_batch_test)

        # Log output predictions in corresponding files
        write_log(message = str(spatial_net_cat_acc.result().numpy()), file_name = spatial_output)
        write_log(message = str(temporal_net_cat_acc.result().numpy()), file_name = temporal_output)
        write_log(message = str(conv_net_cat_acc.result().numpy()), file_name = two_stream_output)

        # Reset states in between iterations: They are averaged after all the testing
        conv_net_cat_acc.reset_state()
        spatial_net_cat_acc.reset_state()
        temporal_net_cat_acc.reset_state()

# Test model
def test_cnn(log_file, model_directory, output, num_iterations = 500,  addition_axis= 0, tensor_num_splits=3, tensor_axis_split=0, type_of_model=3):

    # Set lenght of frames per sequence to test
    len_sequence = 15

    # Load model
    model = models.load_model(model_directory, compile=False)

    # Initialize CNN-RNN function
    feature_extractor = build_feature_extractor()

    # Log starting time of testing process
    write_log(message = "Start of testing at %s" % (str(datetime.now())), file_name = log_file)

    # Iterate testing process
    for iteration in range(num_iterations):

        # Print iteration phase
        write_log(
            "Start of iteration %d of testing"
            % (iteration), log_file
        )

        # Get testing batches
        x_batch_test, y_batch_test = get_frames_sequence(len_sequence=len_sequence, training=False, addition_axis=addition_axis)

        # Define Categorical Accuracy Metrics
        cat_acc = metrics.CategoricalAccuracy()

        # Iterate mini-batches
        for mini_batch in range(len(x_batch_test)):

            # Set mini-batch list values (len = len_sequence = 15)
            x_mini_batch_test = x_batch_test[mini_batch]
            y_mini_batch_test = y_batch_test[mini_batch]

            # Initialize model output list
            output_list = []

            # Split tensor to adjust to model input
            tensor_array = tf.split(x_mini_batch_test, num_or_size_splits=tensor_num_splits, axis=tensor_axis_split)

            # Iterate tensor array and feed it to the model
            for tensor in tensor_array:

                if type_of_model == 3:

                    # Transform input
                    x = feature_extractor(tensor)
                    x = x[None, :]
                    mask = create_batch_mask(5)

                    # Append output value to mini_batch list
                    output_list.append(model((x, mask), training = False)[0])

                elif type_of_model == 4:

                    # Append output value to mini_batch list
                    output_list.append(model(tensor, training = False)[0])

            # Average output list values
            mean_prediction = get_mean_output(output_list)

            # Calculate metrics
            cat_acc.update_state(mean_prediction, y_mini_batch_test)

        # Log output predictions in corresponding files
        write_log(message = str(cat_acc.result().numpy()), file_name = output)

        # Reset states in between iterations: They are averaged after all the testing
        cat_acc.reset_state()

# Output demo directory predictions
def demo_output(image_list, model_directory):

    # Load best model
    model = models.load_model(model_directory, compile=False)
    i = 1
    for image in image_list:
        prediction = model(image, training = False)[0]
        print(
                "Prediction in frame %d: %d"
                % (i, np.argmax(prediction))
            )
            
        i+=1