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
    script_under_test_path = os.path.join(script_folder_path, 'train_model_in_container.py')
    targets_folder_path = os.path.join(script_folder_path, 'mne-side-crf-26-targets')
    model_folder_path = os.path.join(script_folder_path, 'test-train-model-output-model')

    # Run the training
    return_value = subprocess.call([script_under_test_path, targets_folder_path, model_folder_path])
    if return_value != 0:
        raise RuntimeError('Call to %s failed.' % script_under_test_path)
