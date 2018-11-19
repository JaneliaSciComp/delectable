#! /usr/bin/python3

import sys
import os
import tempfile
#import shutil
#import pwd
import dlct
#import moviepy
#import subprocess


def frame_rate_as_rational_string_from_video_file_name(video_file_name) :
    command = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', '0', '-show_entries', 'stream=avg_frame_rate', video_file_name]
    (return_code, stdout_as_string, stderr_as_string) = dlct.system(command)
    if return_code != 0:
        raise RuntimeError(
            'There was a problem running ffprobe to determine the frame rate of %s, return code %d:\n\nstdout: %s\n\nstderr: %s\n' % (
                video_file_name, return_code, stdout_as_string, stderr_as_string))
    return stdout_as_string  # should be a string that looks like a rational number, e.g. '1/1', '30000/1001'


def compress_video(input_video_path,
                   output_video_path):

    # Determine the absolute path to the temp folder that we can write to (e.g. /scratch/svobodalab)
    generalized_slash_tmp_path = dlct.determine_scratch_folder_path()
    print("generalized_slash_tmp_path is %s" % generalized_slash_tmp_path)

    # Get the frame rate of the input video
    frame_rate_as_rational_string = frame_rate_as_rational_string_from_video_file_name(input_video_path)
    print("frame_rate_as_rational_string is %s" % frame_rate_as_rational_string)

    # Make a temporary folder to hold frames, etc
    with tempfile.TemporaryDirectory(prefix=generalized_slash_tmp_path + "/") as scratch_folder_path:
        # Make a folder to hold frames
        frames_folder_path = os.path.join(scratch_folder_path, 'frames')
        os.mkdir(frames_folder_path)

        # Run ffmpeg to convert input video to a folder of .png files
        return_code = os.system('ffmpeg -y -i "%s" "%s/frame-%%06d.png"' % (input_video_path, frames_folder_path))
        if return_code != 0 :
            raise RuntimeError(
                'There was a problem running ffmpeg to convert %s to a folder of frames, return code %d' % (input_video_path, return_code))

        # Use ffmpeg to convert to an HEVC raw bitstream using CRF=35 and with an assumed frame rate of 1 Hz
        return_code = os.system('ffmpeg -y -r 1 -i "%s/frames/frame-%%06d.png" -c:v libx265 -crf 35 -bsf hevc_mp4toannexb "%s/foo-compressed.hevc"'
                                % (scratch_folder_path, scratch_folder_path))
        if return_code != 0 :
            raise RuntimeError(
                'There was a problem running ffmpeg to encode %s using HEVC, return code %d' % (input_video_path, return_code))
        
        # Use mp4box to wrap the raw bitstream in a proper .mp4 container, at the same frame rate as the original
        return_code = os.system('mp4box -add "%s/foo-compressed.hevc" -fps %s "%s"'
                                % (scratch_folder_path, frame_rate_as_rational_string, output_video_path))
        if return_code != 0 :
            raise RuntimeError(
                'There was a problem running mp4box on %s to wrap the HEVC bitstream in a .mp4 container, return code %d' % (input_video_path, return_code))
    

#
# main
#

if __name__ == "__main__" :
    # Get the arguments
    input_video_path = os.path.abspath(sys.argv[1])
    output_video_path = os.path.abspath(sys.argv[2])

    # Run the training
    compress_video(input_video_path, output_video_path)
