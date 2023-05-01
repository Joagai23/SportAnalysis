# Import library
import os
import math
import random
from collections import Counter

# Get father directory where all videos and frames are located
father_directory = "..\Videos"

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
training_file = open("training_list.txt", "x")

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
testing_file = open("testing_list.txt", "x")

# Write test elements in test file
for test in testing_list:

    # Remove file type + father directory path
    test = str(test).replace(".mp4", "").replace("..\Videos\\", "")

    # Write formatted entry + line break
    testing_file.write(test)
    testing_file.write("\n")

# Close training file
testing_file.close()