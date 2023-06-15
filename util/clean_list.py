import os
import shutil

# Set the directory you want to start from
root_dir = 'S:\\server_uploads\\segmentations\\second_batch'

# Loop over all subdirectories in the root_dir
for dir_name in os.listdir(root_dir):
    full_dir_path = os.path.join(root_dir, dir_name)
    if os.path.isdir(full_dir_path):  # make sure it's a directory, not a file
        volumes_dir_path = os.path.join(full_dir_path, 'volume')
        # Check if 'volumes' subdirectory exists and if it's empty
        if os.path.exists(volumes_dir_path) and not os.listdir(volumes_dir_path):
            # The 'volumes' directory is empty
            print(f"Deleting {full_dir_path}...")
            shutil.rmtree(full_dir_path)  # delete the parent directory
