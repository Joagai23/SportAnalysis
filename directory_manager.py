# Import library
import os

# Import dense script
from dense_optical_flow import dense_sequence

# Get father directory where all videos and frames are located
father_directory = "..\Frames"

# Get directory list inside father directory
dir_list = [dirpath for (dirpath, dirnames, filenames) in os.walk(father_directory) for f in filenames]

# Remove duplicates
dir_list = list(dict.fromkeys(dir_list))

# Create sibling list for dense frames
dense_list = []
for directory in dir_list:
    dense_list.append(directory.replace("Frames", "Dense"))

# Call dense function using the directories and dense counterparts as inputs
for directory, dense in zip(dir_list, dense_list):
    #dense_sequence(directory, dense)