# Import libraries
import cv2 as cv
import numpy as np
import os

# Given a video create dense frame sequence in sibling directory
def dense_sequence(video_directory, dense_directory):

    # Read video as Video Capture object
    video_cap = cv.VideoCapture(video_directory)

    # Get first frame of the video
    # ret = boolean value of getting the first frame
    ret, first_frame = video_cap.read()

    # Turn first frame into grayscale (we only need the luminance channel for detecting edges - less computationally expensive)
    previous_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    # Create variable for frame value
    frame_value = 0

    # Iterate until there are no frames left in the video
    while(video_cap.isOpened()):
        
        # Get current frame
        ret, current_frame = video_cap.read()

        # Exit iteration if ret value is false
        if not ret:
            break

        # Convert frame to grayscale
        current_gray = cv.cvtColor(current_frame, cv.COLOR_BGR2GRAY)

        # Calculate dense optical flow using the Farneback method
        optical_flow = cv.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(optical_flow[..., 0], optical_flow[..., 1])

        # Set image hue according to the optical flow direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Set image value according to the optical flow magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Convert HSV to BGR representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        # Check if the directory exists, if not create it
        if not os.path.exists(dense_directory):
            os.makedirs(dense_directory)

        # Save frame in sibling directory
        file_dir = os.path.join(dense_directory, generate_file_name_by_number(frame_value))
        cv.imwrite(file_dir + ".png", rgb)

        # Update frame value
        frame_value += 1

        # Update frames
        previous_gray = current_gray

    # Free video resources
    video_cap.release()

# Transform integer value into 4 char string. Ie.: frame_value = 12 -> ret '0012'
def generate_file_name_by_number(frame_value):
    
    # Get lenght of frame value
    frame_len = len(str(frame_value))

    # Get lenght of zeros before value
    zero_len = 4 - frame_len

    # Create zero string
    zero_string = ""

    # Get zero string
    for i in range(zero_len):
        zero_string = zero_string + '0'

    # Return new formatted string
    return zero_string + str(frame_value)