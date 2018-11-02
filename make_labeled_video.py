"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script labels the bodyparts in videos as analzyed by "AnalyzeVideos.py". This code is relatively slow as 
it stores all individual frames. 

Use  MakingLabeledVideo_fast.py instead for faster (and slightly different) version (frames are not stored).

python3 MakingLabeledVideo.py

Note: run python3 AnalyzeVideos.py first.
"""

# This is a modified version of raw DLC's Analysis-tools/MakingLabeledVideo.py

####################################################
# Dependencies
####################################################

import os
import sys

# Add folders to the python path
path_to_this_script = os.path.abspath(__file__)
path_to_delectable_folder = os.path.dirname(path_to_this_script)
dlc_folder_path = os.path.join(path_to_delectable_folder, "dlc")
sys.path.append(os.path.join(dlc_folder_path, "pose-tensorflow"))
sys.path.append(os.path.join(dlc_folder_path, "Generating_a_Training_Set"))

# subfolder = os.getcwd().split('Analysis-tools')[0]
# sys.path.append(subfolder)
# # add parent directory: (where nnet & config are!)
# sys.path.append(subfolder + "/pose-tensorflow/")
# sys.path.append(subfolder + "/Generating_a_Training_Set")

# Dependencies for video:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
#imageio.plugins.ffmpeg.download()
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob
#import auxiliaryfunctions
import dlct
import tempfile
import shutil
import time

####################################################
# Loading descriptors of model
####################################################

#from myconfig_analysis import videofolder, cropping, Task, date, \
#    resnet, shuffle, trainingsiterations, pcutoff, deleteindividualframes, x1, x2, y1, y2, videotype, alphavalue, dotsize, colormap

#from myconfig_analysis import scorer as humanscorer

# Get the command-line args
model_folder_path = os.path.abspath(sys.argv[1])
video_file_path = os.path.abspath(sys.argv[2])
h5_file_path = os.path.abspath(sys.argv[3])

# Load the configuration file
configuration_file_name = 'myconfig.py'
configuration_file_path = os.path.join(model_folder_path, configuration_file_name)
configuration = dlct.load_configuration_file(configuration_file_path)
Task = configuration['Task']
date = configuration['date']
#trainingsFraction = configuration['trainingsFraction']
resnet = configuration['resnet']
snapshotindex = configuration['snapshotindex']
#shuffle = configuration['shuffle']
cropping = configuration['cropping']
x1 = configuration['x1']
x2 = configuration['x2']
y1 = configuration['y1']
y2 = configuration['y2']
#videotype = configuration['videotype']
#storedata_as_csv = configuration['storedata_as_csv']
trainingFractionList = configuration['TrainingFraction']
shuffleList = configuration['Shuffles']
#humanscorer = configuration['scorer']

# These things are in myconfig_analysis.py in raw DeepLabCut
trainingFraction = trainingFractionList[0]
shuffle = shuffleList[0]
pcutoff = 0.1
#deleteindividualframes = False
alphavalue=1 # "strength/transparency level of makers" in individual frames (Vary from 0 to 1. / not working in "MakingLabeledVideo_fast.py")
dotsize = 7
colormap_name = 'Set1' #other colorschemes: 'cool' and see https://matplotlib.org/examples/color/colormaps_reference.html

# loading meta data / i.e. training & test files
#basefolder = os.path.join('..','pose-tensorflow','models')
#datafolder = os.path.join(basefolder , "UnaugmentedDataSet_" + Task + date)
#Data = pd.read_hdf(os.path.join(datafolder , 'data-' + Task , 'CollectedData_' + humanscorer + '.h5'),'df_with_missing')

# Name for scorer based on passed on parameters from myconfig_analysis. Make sure they refer to the network of interest.
#scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)

####################################################
# Auxiliary function
####################################################

# # https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
# def get_repeated_cmap(cmap_name, n_parts):
#     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
#     RGB color; the keyword argument name must be a standard mpl colormap name.'''
#     #return plt.cm.get_cmap(cmap_name, n)
#
#     raw_cmap = plt.cm.get_cmap(cmap_name)
#     n_raw_colors = raw_cmap.N
#
#     def cmap(i) :
#         return raw_cmap(i % n_raw_colors)
#
#     return cmap


def CreateVideo(clip, Dataframe, pcutoff, colormap_name, alphavalue, frames_folder_path, output_file_path):
    ''' Creating individual frames with labeled body parts and making a video''' 
    scorer=np.unique(Dataframe.columns.get_level_values(0))[0]
    bodyparts2plot = list(np.unique(Dataframe.columns.get_level_values(1)))
    colors = dlct.get_repeated_cmap(colormap_name, len(bodyparts2plot))

    ny, nx = clip.size  # dimensions of frame (height, width)
    fps = clip.fps
    nframes = len(Dataframe.index)
    if cropping:
        # one might want to adjust
        clip = clip.crop(y1=y1, y2=y2, x1=x1, x2=x2)
    clip.reader.initialize()
    print("Duration of video [s]: ", clip.duration, ", recorded with ", fps,
          "fps!")
    print("Overall # of frames: ", nframes, "with cropped frame dimensions: ",
          clip.size)
    print("Generating frames")
    for index in tqdm(range(nframes)):

        frame_file_path = os.path.join(frames_folder_path, "frame-%04d.png" % index)
        if os.path.isfile(frame_file_path):
            pass
        else:
            plt.axis('off')
            image = img_as_ubyte(clip.reader.read_frame())
            #image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))

            if np.ndim(image) > 2:
                h, w, nc = np.shape(image)
            else:
                h, w = np.shape(image)

            plt.figure(frameon=False, figsize=(w * 1. / 100, h * 1. / 100))
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.imshow(image)

            for bpindex, bp in enumerate(bodyparts2plot):
                if Dataframe[scorer][bp]['likelihood'].values[index] > pcutoff:
                    plt.scatter(
                        Dataframe[scorer][bp]['x'].values[index],
                        Dataframe[scorer][bp]['y'].values[index],
                        s=dotsize**2,
                        color=colors(bpindex),
                        alpha=alphavalue)

            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.axis('off')
            plt.subplots_adjust(
                left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            plt.gca().invert_yaxis()
            plt.savefig(frame_file_path)
            plt.clf()

            plt.close("all")

    #os.chdir(frames_folder_path)

    print("All labeled frames were created, now generating video...")
    try:
        #subprocess.call([
        #    'ffmpeg', '-y', '-framerate',
        #    str(clip.fps), '-i', 'frame-%04d.png', '-pix_fmt', 'yuv420p', output_file_path])
        subprocess.call([
            'ffmpeg', '-y', '-framerate',
            '30', '-i', os.path.join(frames_folder_path, 'frame-%04d.png'), '-pix_fmt', 'yuv420p', output_file_path])
    except FileNotFoundError:
        print("Ffmpeg not correctly installed, see https://github.com/AlexEMG/DeepLabCut/issues/45")

    # if deleteindividualframes:
    #     for file_name in glob.glob("*.png"):
    #         os.remove(file_name)
    #
    # os.chdir("../")

##################################################
# Datafolder
##################################################
# videofolder='../videos/' #where your folder with videos is.

#os.chdir(videofolder)
#videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn) and ("labeled" not in fn)])

#print("Starting ", videofolder, videos)
#for video in videos:
#video_file_path = video_file_path
#vname = video.split('.')[0]
#tmpfolder = 'temp' + vname
#auxiliaryfunctions.attempttomakefolder(tmpfolder)

generalized_slash_tmp_path = dlct.determine_scratch_folder_path()
print("generalized_slash_tmp_path is %s" % generalized_slash_tmp_path)
scratch_folder_path_maybe = []  # want to keep track of this so we know whether or not to delete it

# Synthesize the output file path
output_file_path = dlct.replace_extension(h5_file_path, '.mp4')

# Make a temporary folder
#scratch_folder_path = dlct.replace_extension(h5_file_path, '-frames')
scratch_folder_path = tempfile.mkdtemp(prefix=generalized_slash_tmp_path + "/")
with tempfile.TemporaryDirectory(prefix=generalized_slash_tmp_path + "/") as scratch_folder_path :
    print("scratch_folder_path is %s" % scratch_folder_path)

    # Make the video with the little dots in it
    #if os.path.exists(output_file_path):
    #    print("Labeled video already created.")
    #else:
    print("Loading ", video_file_path, "and data.")
    #h5_file_path = h5_file_path
    Dataframe = pd.read_hdf(h5_file_path)
    clip = VideoFileClip(video_file_path)
    CreateVideo(clip, Dataframe, pcutoff, colormap_name, alphavalue, scratch_folder_path, output_file_path)

    #time.sleep(1)

    #scratch_folder_path.cleanup()

