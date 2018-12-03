#! /usr/bin/env python3

import sys
import os
import subprocess
import shutil

#
# main
#
if __name__ == "__main__":
    # Get the arguments
    script_path = os.path.realpath(__file__)
    script_folder_path = os.path.dirname(script_path)
    script_under_test_path = os.path.join(script_folder_path, 'compress_videos_on_cluster.py')
    read_only_input_folder_path = os.path.join(script_folder_path, 'videos-to-compress-read-only')
    input_folder_path = os.path.join(script_folder_path, 'videos-to-compress')
    output_folder_path = os.path.join(script_folder_path, 'videos-that-are-compressed')

    # Set up the test folders
    if os.path.exists(input_folder_path):    
        shutil.rmtree(input_folder_path)
    shutil.copytree(read_only_input_folder_path, input_folder_path)

    if os.path.exists(output_folder_path):    
        shutil.rmtree(output_folder_path)
    os.mkdir(output_folder_path)

    # Run the training
    return_value = subprocess.call([script_under_test_path, input_folder_path, output_folder_path])
    if return_value != 0:
        raise RuntimeError('Call to %s failed.' % script_under_test_path)
