# Import library
import os

# Import dense script
from dense_optical_flow import dense_sequence

# Get father directory where all videos and frames are located
father_directory = "..\Videos"

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