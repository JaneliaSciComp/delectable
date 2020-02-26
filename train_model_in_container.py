#! /usr/bin/env python3

import sys
import os
import subprocess



#
# main
#
if __name__ == "__main__":
    # Get the arguments
    script_path = os.path.realpath(__file__)
    script_folder_path = os.path.dirname(script_path)
    script_under_test_path = os.path.join(script_folder_path, 'train_model.py')
    targets_folder_path = sys.argv[1]
    model_folder_path = sys.argv[2]
    singularity_image_path = os.path.join(script_folder_path, "dlc.simg")
    paths_to_mount_in_container = "/tmp,"+script_folder_path
    
    # Run the training
    return_value = subprocess.call(["singularity", "exec", "-B", paths_to_mount_in_container, "--nv", singularity_image_path,
                                    "python3", script_under_test_path, targets_folder_path, model_folder_path])
    if return_value != 0:
        raise RuntimeError('Call to %s failed.' % script_under_test_path)
