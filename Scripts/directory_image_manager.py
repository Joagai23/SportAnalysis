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

# Define directories
video_directory = "../Videos"
frame_directory = "SportAnalysis/Frames"
dense_directory = "SportAnalysis/Dense"

# Define files
training_file_directory = "SportAnalysis/Text_Files/training_directory_list.txt"
testing_file_directory = "SportAnalysis/Text_Files/testing_directory_list.txt"
temporal_model_log = "SportAnalysis/Text_Files/temporal_model.txt"
spatial_model_log = "SportAnalysis/Text_Files/spatial_model.txt"

# Define model directories
temporal_model_directory = "SportAnalysis/Models/temporal_model"
spatial_model_directory = "SportAnalysis/Models/spatial_model"

# For every video create a dense optical flow
def dense_flow():

    # Get directory list inside father directory
    dir_list = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(video_directory) for f in filenames]

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
    dir_list = [dirpath for (dirpath, dirnames, filenames) in os.walk(video_directory) for f in filenames]

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
    training_file = open(training_file_directory, "x")

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
    testing_file = open(testing_file_directory, "x")

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

# Get a random dense sequence from a video directory
def __get_random_dense_frame_and_label(directory, frame_dir, lenght_sequence = 3):

    # Fix string termination
    directory = str(directory).replace("\n", "")
    
    # Join directory paths and get a random frame from it
    pathList = os.listdir(frame_dir + directory)

    # Get frame that has a sequence (not end of video)
    frame = random.choice(pathList[:-lenght_sequence])

    # Get frame position in order to get sequence
    position = pathList.index(frame)

    # Add consecutive frames to list of names
    frame_list = [str(frame_dir + directory + "/" + frame)]
    for i in range(1, lenght_sequence):
        frame_list.append(str(frame_dir + directory + "/" + pathList[position + i]))

    # Return new list file path
    return frame_list, str(directory).split("/")[1]

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

# Transform path list into dense matrix array
def dense_path_to_image(image_path_matrix):

    # Define output list and image dimensions
    image_matrix = []
    image_size = (224, 224)

    # Iterate paths and transform into images of the right size
    for image_path_list in image_path_matrix:
        image_list = []
        for image_path in image_path_list:
            image = cv.imread(image_path, cv.IMREAD_COLOR)
            image = cv.resize(image, image_size)
            if not len(image_list):
                image_list = image[None, :]
            else:
                image_list = np.concatenate((image_list, image[None, :]), axis=3)
        image_matrix.append(image_list)

    return image_matrix

# Testing flow of transforming a list of dense files into list of matrices
def test_dense_path_to_image(image_path_list, len_aggroupation = 3, len_list_ouput = 15):

    # Define output list, image dimensions and aggroupation list
    aggroupation_array = []

    # Group image path list into chunks of lenght = len_aggroupation
    for current_pos in range(0, len_list_ouput):
        temporal_list = []
        for i in range(current_pos, len_aggroupation + current_pos):
            temporal_list.append(image_path_list[i])
        aggroupation_array.append(temporal_list)

    return dense_path_to_image(aggroupation_array)

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

# Transform string label into one-hot encoding
def string_label_to_binary(label):

    # Define numeric value: by default 'transition'
    number_value = 3

    # Transform string into numeric value
    if label == 'equality':
            number_value = 0
    elif label == 'penalty':
        number_value = 1
    elif label == 'superiority':
        number_value = 2

    # Transform and return
    return utils.to_categorical(number_value, num_classes=4)

# Get batch of training data for one iteration
def get_training_data(type_of_model = 1):

    if type_of_model == 2:
        return get_dense_training_data()

    # Open and read training file
    training_file = open(training_file_directory, "r")

    # Define train data
    x_batch_train = []
    y_batch_train = []

    # Iterate lines in training file
    for line in training_file:

        # For every line obtain random frame with label
        frame, label = __get_random_frame_and_label(line, frame_directory)
        x_batch_train.append(frame)
        y_batch_train.append(label)

    # Return file pointer to starting position and close it
    training_file.seek(0)
    training_file.close()
    
    # Return copies of training batches
    return path_to_image(x_batch_train), transform_labels_to_number(y_batch_train)

# Get batch of dense training data for one iteration
def get_dense_training_data():

    # Open and read training file
    training_file = open(training_file_directory, "r")

    # Define train data
    x_batch_train = []
    y_batch_train = []

    # Iterate lines in training file
    for line in training_file:

        # For every line obtain random frame with label
        frame_list, label = __get_random_dense_frame_and_label(line, dense_directory)
        x_batch_train.append(frame_list)
        y_batch_train.append(label)

    # Return file pointer to starting position and close it
    training_file.seek(0)
    training_file.close()

    # Return copies of training batches
    return dense_path_to_image(x_batch_train), transform_labels_to_number(y_batch_train)

# Return list of file paths 
def find_file_sequence_by_dense(dense_dir, video_path, len_sequence = 15):
    
    # Fix string termination
    directory = str(video_path).replace("\n", "")

    # Get list of frames from directory
    pathList = os.listdir(dense_dir + directory)

    # Get frame that has a sequence (not end of video)
    frame = random.choice(pathList[:-len_sequence])
    
    # Get frame position inside list
    position = pathList.index(frame)

    # Add consecutive frames to list of names
    frame_list = [str(directory + "/" + frame)]
    for i in range(1, len_sequence):
        frame_list.append(str(directory + "/" + pathList[position + i]))

    return frame_list

# Get a sequence of mirror frames and dense
def get_test_frames_dense(len_sequence = 15):

    # Open and read training file
    training_file = open(training_file_directory, "r")

    # Define train data for both spatial and temporal flows
    spatial_x_batch_test = []
    temporal_x_batch_test = []
    y_batch_test = []

    # Dense directory lenght = Frame directory lenght - 1
    # Dense determines que files to use -> then they are mapped to frames
    # Dense inputs are 2 files more than Frame inputs
    dense_len_sequence = len_sequence + 2
    len_dense_aggroupation = 3

    # Iterate lines in training file
    for line in training_file:

        # Find sequence of frames
        file_sequence = find_file_sequence_by_dense(dense_dir, line, dense_len_sequence)
        
        # Get label for current video path
        label = str(line).split("/")[1]

        # Process label into one-hot and add it to label array
        label = string_label_to_binary(label)
        y_batch_test.append(label)

        # Get spatial input
        spatial_frames = [frame_directory + file for file in file_sequence[:len_sequence]]
        spatial_x_batch_test.append(path_to_image(spatial_frames))

        # Get temporal input
        temporal_frames = [dense_directory + file for file in file_sequence]
        temporal_x_batch_test.append(test_dense_path_to_image(temporal_frames, len_dense_aggroupation, len_sequence))

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