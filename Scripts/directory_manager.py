# Import library
import os
import math
import random
import numpy as np
import cv2 as cv
import re
from collections import Counter
from keras import utils

# Import dense script
from dense_optical_flow import dense_sequence

# Get father directory where all videos and frames are located
father_directory = "..\Videos"

# For every video create a dense optical flow
def dense_flow():

    # Get directory list inside father directory
    dir_list = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(father_directory) for f in filenames]

    # Remove duplicates
    dir_list = list(dict.fromkeys(dir_list))

    # Create sibling list for dense frames
    dense_list = []
    for directory in dir_list:
        dense_list.append(directory.replace("Videos", "Dense"))

    # Create variables to control process
    number_dir = len(dir_list)
    current_dir = 0

    # Call dense function using the video path and dense counterpart as inputs
    for directory, dense in zip(dir_list, dense_list):
        
        # Update and print progress
        current_dir += 1
        print("Dense progress:", str((current_dir / number_dir) * 100), "%")

        dense_sequence(directory, str(dense).replace(".mp4", ""))

# Create files containing the directories of entries used for training and testing
def create_test_train_files():

    # Get directory list inside father directory
    dir_list = [dirpath for (dirpath, dirnames, filenames) in os.walk(father_directory) for f in filenames]

    # Get ocurrences of every type of video
    count = Counter(dir_list)

    # For every type of video 80% on entries go to TRAINING dataset and 20% to TESTING dataset
    training_rate = 0.8
    testing_rate = 0.2

    # Show ocurrence distribution
    for key in count:
        print(key, '->', count[key])
        print(key, '->', 'train:', math.floor(count[key] * training_rate))
        print(key, '->', 'test:', math.ceil(count[key] * testing_rate))

    # Create list of test and train items
    training_list = []
    testing_list = []

    # Iterate ocurrence distribution
    for key in count:

        # Get number of entries that need to go to each list
        training_num = math.floor(count[key] * training_rate)
        testing_num = math.ceil(count[key] * testing_rate)

        # Get files in every key directory
        dir_names = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(key) for f in filenames]

        # Fill testing list
        for i in range(training_num):
            train_entry = random.choice(dir_names)
            training_list.append(train_entry)
            dir_names.remove(train_entry)

        # Fill training list
        testing_list.extend(dir_names)

    # Create and open training file
    training_file = open("training_directory_list.txt", "x")

    # Write train elements in training file
    for train in training_list:

        # Remove file type + father directory path
        train = str(train).replace(".mp4", "").replace("..\Videos\\", "")

        # Write formatted entry + line break
        training_file.write(train)
        training_file.write("\n")

    # Close training file
    training_file.close()

    # Create and open testing file
    testing_file = open("testing_directory_list.txt", "x")

    # Write test elements in test file
    for test in testing_list:

        # Remove file type + father directory path
        test = str(test).replace(".mp4", "").replace("..\Videos\\", "")

        # Write formatted entry + line break
        testing_file.write(test)
        testing_file.write("\n")

    # Close training file
    testing_file.close()

# Get a random frame from a video directory
def __get_random_frame_and_label(directory, frame_dir):

    # Fix string termination
    directory = str(directory).replace("\n", "")
    
    # Join directory paths and get a random frame from it
    pathList = os.listdir(frame_dir + directory)
    frame = random.choice(pathList)

    # Return new file path
    return str(frame_dir + directory + "/" + frame), str(directory).split("/")[1]

# Shuffle two lists of the same lenght
def __unison_shuffled_copies(list_1, list_2):

    # Make sure they are the same lenght
    assert len(list_1) == len(list_2)

    # Create random permutations for a list of the same lenght
    p = np.random.permutation(len(list_1))

    # Return shuffled lists
    return list_1[p], list_2[p]

# Transform path list into image matrix array
def path_to_image(image_path_list):

    # Define output list and image dimensions
    image_list = []
    image_size = (224, 224)

    # Iterate paths and transform into images of the right size
    for image_path in image_path_list:
        image = cv.imread(image_path, cv.IMREAD_COLOR)
        image = cv.resize(image, image_size)
        image_list.append(image[None, :])

    return image_list

# Transform list of string outputs into categorical output. Ie.: ['equality','penalty','superiority','transition'] -> [[1 0 0 0][0 1 0 0][0 0 1 0][0 0 0 1]]
def transform_labels_to_number(label_list):

    # Create list of numbers
    number_list = []

    # Iterate labels
    for label in label_list:
        
        if label == 'equality':
            number_list.append(0)
        elif label == 'penalty':
            number_list.append(1)
        elif label == 'superiority':
            number_list.append(2)
        else:
            number_list.append(3)

    # Transform and return categorical list
    return utils.to_categorical(number_list, num_classes=4)

# Get batch of training data for one iteration
def get_training_data():

    # Define frame directory
    frame_dir = "SportAnalysis/Frames"

    # Open and read training file
    training_file = open("SportAnalysis/Text_Files/training_directory_list.txt", "r")

    # Define train data
    x_batch_train = []
    y_batch_train = []

    # Iterate lines in training file
    for line in training_file:

        # For every line obtain random frame with label
        frame, label = __get_random_frame_and_label(line, frame_dir)
        x_batch_train.append(frame)
        y_batch_train.append(label)

    # Return file pointer to starting position and close it
    training_file.seek(0)
    training_file.close()
    
    # Return copies of training batches
    return path_to_image(x_batch_train), transform_labels_to_number(y_batch_train)

# Rename directory from /goal (19) ---> to /goal_19
def rename_directory(father_directory):

    # Get directory list inside father directory
    dir_list = [dirpath for (dirpath, dirnames, filenames) in os.walk(father_directory) for f in filenames]

    # Remove duplicates
    dir_list = list(dict.fromkeys(dir_list))

    # Iterate list of directories inside main directory
    for directory in dir_list:

        # Get number value inside directory
        number = int(re.findall(r'\d+', directory)[0])

        # Get new directory path
        new_directory = directory.replace(" (%d)" % (number), "_%d" % (number))

        # Replace directory and pray
        os.rename(directory, new_directory)