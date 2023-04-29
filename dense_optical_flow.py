# Import libraries
import cv2 as cv
import numpy as np
import os
from matplotlib import image

example_frame_directory = "..\Frames\equality\goal\equality_goal (2)"
example_dense_directory = "..\Dense\equality\goal\equality_goal (2)"

# Given a directory create dense frame sequence in sibling directory
def dense_sequence(frame_directory, dense_directory):

    # Get list of files (frames) in directory with full path
    file_list = os.listdir(frame_directory)

    # Get first frame name in list and remove it
    previous_frame_name = file_list.pop(0)

    # Convert first frame name into image and array
    previous_frame = np.asarray(image.imread(os.path.join(frame_directory, previous_frame_name)))

    # Turn first frame into grayscale (we only need the luminance channel for detecting edges - less computationally expensive)
    previous_gray = cv.cvtColor(previous_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(previous_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    # Iterate until directory list is empty
    while(file_list):
        
        # Get current frame name in iteration
        current_frame_name = file_list.pop(0)

        # Convert current frame name into image and array
        current_frame = np.asarray(image.imread(os.path.join(frame_directory, current_frame_name)))

        # Convert frame to grayscale
        current_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

        # Calculate dense optical flow using the Farneback method
        optical_flow = cv.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])

        # Sets image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to BGR representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        # Check if the directory exists, if not create it
        if not os.path.exists(dense_directory):
            os.makedirs(dense_directory)

        # Save frame in sibling directory
        file_dir = os.path.join(dense_directory, current_frame_name)
        cv.imwrite(file_dir, rgb)

        # Update frames
        previous_gray = current_gray

dense_sequence(example_frame_directory, example_dense_directory)