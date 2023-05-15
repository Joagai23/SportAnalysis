# Append message to log file
def write_log(message, file_name):

    # Open file in append mode
    file = open(file=file_name, mode="a")

    # Write log
    file.write(message)
    
    # Write line jump
    file.write("\n")

    # Close file
    file.close()