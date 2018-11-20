#! /usr/bin/python3

import sys
import os
import tempfile
import dlct
#import shutil
from compress_video import compress_video


# def frame_rate_as_rational_string_from_video_file_name(video_file_name) :
#     # Use ffprobe to get the LCM frame rate of a video.
#     # These LCM frame rates seem to be better preserved than the average frame rate.
#     command = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', '0', '-show_entries', 'stream=r_frame_rate', video_file_name]
#     (return_code, stdout_as_string, stderr_as_string) = dlct.system(command)
#     if return_code != 0:
#         raise RuntimeError(
#             'There was a problem running ffprobe to determine the frame rate of %s, return code %d:\n\nstdout: %s\n\nstderr: %s\n' % (
#                 video_file_name, return_code, stdout_as_string, stderr_as_string))
#     return stdout_as_string.strip()  # should be a string that looks like a rational number, e.g. '1/1', '30000/1001'


def compress_video_and_delete_lock_file(input_video_path,
                                        lock_file_path,
                                        output_video_path):
    try:
        compress_video(input_video_path, output_video_path)
    except Exception as e:
        os.remove(lock_file_path)
        raise e
    os.remove(lock_file_path)


#
# main
#

if __name__ == "__main__" :
    # Get the arguments
    input_video_path = os.path.abspath(sys.argv[1])
    lock_file_path = os.path.abspath(sys.argv[2])
    output_video_path = os.path.abspath(sys.argv[3])
    compress_video_and_delete_lock_file(input_video_path,
                                        lock_file_path,
                                        output_video_path)
