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
    script_under_test_path = os.path.join(script_folder_path, 'make_labeled_video_on_cluster')
    model_folder_path = os.path.join(script_folder_path, 'mne-side-crf-26-model-read-only')
    video_file_path = os.path.join(script_folder_path, 'mne-side-crf-26-test-video.mp4')
    h5_file_path = os.path.join(script_folder_path, 'mne-side-crf-26-test-video-read-only.h5')
    output_video_file_path = os.path.join(script_folder_path, 'test-make-labeled-video-output.mp4')

    # Run the training
    return_value = subprocess.call([script_under_test_path, model_folder_path, video_file_path, h5_file_path, output_video_file_path])
    if return_value != 0:
        raise RuntimeError('Call to %s failed.' % script_under_test_path)
