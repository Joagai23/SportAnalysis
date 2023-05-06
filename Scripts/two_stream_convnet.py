# Import libraries
import tensorflow as tf
import time
from keras import optimizers, losses, metrics

# Import functions
from spatial_stream_conv import create_spatial_model
from directory_manager import get_training_data
from log_writer import write_log

# Define model variables
image_size = (224, 224)
dropout_value = 0.5
num_outputs = 4
epochs = [50000, 20000, 10000]
learning_rates = [1e-2, 1e-3, 1e-4]

# Create spatial model
spatial_model =  create_spatial_model(image_size + (3,), num_classes=4, dropout_value=dropout_value)

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
        % (epoch, learn_rate), "SportAnalysis/Text_Files/spatial_model.txt"
    )

    # Get epoch starting time
    start_time = time.time()

    # Iterate batches
    for step in range(epoch):

        # Obtain training inputs and outputs
        x_batch_train, y_batch_train = get_training_data()

        # Feed inputs to model
        for x, y in zip(x_batch_train, y_batch_train):

            # Record operations with Gradient Tape
            with tf.GradientTape() as tape:

                # Forward pass
                prediction = spatial_model(x, training = True)

                # Obtain tensor value from shape=(1, 4) to shape(4,)
                prediction = prediction[0]

                # Compute loss value
                loss_value = loss_function(y, prediction)

            # Retrieve gradients of trainable variables
            gradients = tape.gradient(loss_value, spatial_model.trainable_weights)

            # Run gradient descent step
            optimizer.apply_gradients(zip(gradients, spatial_model.trainable_weights))

            # Update training metric
            training_metrics.update_state(y, prediction)

        # Log every 500 iterations
        if step % 500 == 0 and step != 0:
            write_log(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value)), "SportAnalysis/Text_Files/spatial_model.txt"
            )

            # Save model
            spatial_model.save("SportAnalysis/Models/spatial_model_%d_%d" % (epoch, step))

    # Reset training metrics at the end of each epoch
    training_metrics.reset_states()

    # Show training time for every epoch
    write_log("Time taken: %.2fs" % (time.time() - start_time), "SportAnalysis/Text_Files/spatial_model.txt")