#! /usr/bin/python3

import sys
import os
import tempfile
import dlct
import shutil


def frame_rate_as_rational_string_from_video_file_name(video_file_name) :
    # Use ffprobe to get the LCM frame rate of a video.
    # These LCM frame rates seem to be better preserved than the average frame rate.
    command = ['ffprobe', '-v', '0', '-of', 'csv=p=0', '-select_streams', '0', '-show_entries', 'stream=r_frame_rate', video_file_name]
    (return_code, stdout_as_string, stderr_as_string) = dlct.system(command)
    if return_code != 0:
        raise RuntimeError(
            'There was a problem running ffprobe to determine the frame rate of %s, return code %d:\n\nstdout: %s\n\nstderr: %s\n' % (
                video_file_name, return_code, stdout_as_string, stderr_as_string))
    return stdout_as_string.strip()  # should be a string that looks like a rational number, e.g. '1/1', '30000/1001'


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
        print("scratch_folder_path is %s" % scratch_folder_path)

        # Make a folder to hold frames
        frames_folder_path = os.path.join(scratch_folder_path, 'frames')
        os.mkdir(frames_folder_path)

        # Run ffmpeg to convert input video to a folder of .png files
        #return_code = os.system('ffmpeg -y -i "%s" "%s/frame-%%06d.png"' % (input_video_path, frames_folder_path))
        command = ['ffmpeg', '-y', '-i', input_video_path, os.path.join(frames_folder_path, 'frame-%06d.png') ]
        (return_code, stdout_as_string, stderr_as_string) = dlct.system(command)
        if return_code != 0 :
            raise RuntimeError(
                'There was a problem running ffmpeg to convert %s to a folder of frames, return code %d.\n\n stdout:\n%s\n\nstderr:\n%s\n'
                % (input_video_path, return_code, stdout_as_string, stderr_as_string))

        # Use ffmpeg to convert to an HEVC raw bitstream using CRF=35 and with an assumed frame rate of 1 Hz
        scratch_bitstream_file_path = os.path.join(scratch_folder_path, 'foo-compressed.hevc')
        command = ['ffmpeg', '-y', '-framerate', '1', '-i', os.path.join(scratch_folder_path, 'frames', 'frame-%06d.png'),
                   '-pix_fmt', 'yuv420p', '-c:v', 'libx265',  '-crf', '35', '-bsf', 'hevc_mp4toannexb',
                   scratch_bitstream_file_path ]
        (return_code, stdout_as_string, stderr_as_string) = dlct.system(command)
        # return_code = os.system('ffmpeg -y -r 1 -i "%s/frames/frame-%%06d.png" -c:v libx265 -crf 35 -bsf hevc_mp4toannexb "%s/foo-compressed.hevc"'
        #                         % (scratch_folder_path, scratch_folder_path))
        if return_code != 0 :
            raise RuntimeError(
                'There was a problem running ffmpeg to encode %s using HEVC, return code %d.\n\nstdout:\n%s\n\nstderr:\n%s\n'
                % (input_video_path, return_code, stdout_as_string, stderr_as_string))

        # # Do the same thing, but wrap the result in a .mp4 container, for testing purposes
        # command = ['ffmpeg', '-y', '-framerate', '1', '-i', os.path.join(scratch_folder_path, 'frames', 'frame-%06d.png'),
        #            '-pix_fmt', 'yuv420p', '-c:v', 'libx265',  '-crf', '35',
        #            os.path.join(scratch_folder_path, 'foo-compressed-1-hz.hevc.mp4') ]
        # (return_code, stdout_as_string, stderr_as_string) = dlct.system(command)
        # # return_code = os.system('ffmpeg -y -r 1 -i "%s/frames/frame-%%06d.png" -c:v libx265 -crf 35 -bsf hevc_mp4toannexb "%s/foo-compressed.hevc"'
        # #                         % (scratch_folder_path, scratch_folder_path))
        # if return_code != 0 :
        #     raise RuntimeError(
        #         'There was a problem running ffmpeg to encode %s using HEVC (testing part), return code %d.\n\nstdout:\n%s\n\nstderr:\n%s\n'
        #         % (input_video_path, return_code, stdout_as_string, stderr_as_string))

        # # convert the (rational) frame rate to a double, and thence to a string
        # numerator_and_denominator = frame_rate_as_rational_string.split('/')
        # if len(numerator_and_denominator) == 0:
        #     raise RuntimeError(
        #         'The frame rate of the file %s, as returned by ffprobe, does not seem to be valid.  It is "%s".'
        #         % (input_video_path, frame_rate_as_rational_string))
        # elif len(numerator_and_denominator) == 1:
        #     numerator = numerator_and_denominator[0]
        #     frame_rate_as_double = float(numerator)
        # else:
        #     numerator = numerator_and_denominator[0]
        #     denominator = numerator_and_denominator[1]
        #     frame_rate_as_double = float(numerator)/float(denominator)
        # print('frame_rate_as_double: %s' % repr(frame_rate_as_double))

        # Use ffmpeg to wrap the raw bitstream in a proper .mp4 container, at the same LCM frame rate as the original
        # mp4box apparently works better if your inputs have b-pyramids, but that's another dependency, and
        # I'm hoping these videos won't have b-pyrmaids, since they are all outputs of ffmpeg that have passed through
        # the bottleneck of being represented as a folder of frames.
        #scratch_output_video_path = os.path.join(scratch_folder_path, 'foo-compressed.hevc.mp4')
        command = ['ffmpeg', '-y', '-r', frame_rate_as_rational_string, '-i', scratch_bitstream_file_path, '-c', 'copy', output_video_path]
        # Note that using -framerate instead of -r doesn't work: If you do this, the output is at 1 Hz.
        # At least for ffmpeg 4.0.2...
        (return_code, stdout_as_string, stderr_as_string) = dlct.system(command)
        if return_code != 0 :
            raise RuntimeError(
                'There was a problem running mp4box on %s to wrap the HEVC bitstream in a .mp4 container, return code %d.\n\nstdout:\n%s\n\nstderr:\n%s\n'
                % (input_video_path, return_code, stdout_as_string, stderr_as_string))

        # # Copy the final video to the output path
        # shutil.copyfile(scratch_output_video_path, output_video_path)

        # Check the frame rate of the output
        output_frame_rate_as_rational_string = frame_rate_as_rational_string_from_video_file_name(output_video_path)
        print("output_frame_rate_as_rational_string is %s" % output_frame_rate_as_rational_string)


#
# main
#

if __name__ == "__main__" :
    # Get the arguments
    input_video_path = os.path.abspath(sys.argv[1])
    if len(sys.argv) >= 3:
        output_video_path = os.path.abspath(sys.argv[2])
    else:
        output_video_path = dlct.replace_extension(input_video_path, '.hevc.mp4')

    # Run the training
    compress_video(input_video_path, output_video_path)
