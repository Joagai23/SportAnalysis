# Import libraries
import tensorflow as tf
import numpy as np
from keras import optimizers, losses

# Import functions
from spatial_stream_conv import create_spatial_model
from directory_manager import get_training_data

# Define model variables
image_size = (224, 224)
dropout_value = 0.5
num_outputs = 4
epochs = [50000, 20000, 10000]
learning_rates = [1e-2, 1e-3, 1e-4]

# Create spatial model
spatial_model =  create_spatial_model(image_size + (3,), num_classes=4, dropout_value=dropout_value)

# Instantiate loss function
loss_function = losses.CategoricalCrossentropy(from_logits=True)

# Zip epoch and learning rate pair and iterate
for epoch, learn_rate in zip(epochs, learning_rates):
    
    # Instantiate optimizer
    optimizer = optimizers.SGD(learning_rate=learn_rate)

    # Print Iteration Phase
    print(
        "\nStart of %d iterations with learning rate %.4f"
        % (epoch, learn_rate)
    )

    # Iterate batches
    for step in range(epoch):

        # Obtain training inputs and outputs
        x_batch_train, y_batch_train = get_training_data()

        # Record operations with Gradient Tape
        with tf.GradientTape() as tape:

            # Forward pass
            logits = spatial_model(x_batch_train, training = True)

            # Compute loss
            loss_value = loss_function(y_batch_train, logits)

        # Retrieve gradients of trainable variables
        gradients = tape.gradient(loss_value, spatial_model.trainable_weights)

        # Run gradient descent step
        optimizer.apply_gradients(zip(gradients, spatial_model.trainable_weights))

        # Log every 500 iterations
        if step % 500 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )

            # Save model
            spatial_model.save("./model_save/spatial_model_%d_%d" % (epoch, step))