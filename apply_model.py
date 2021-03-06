"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script analyzes videos based on a trained network (as specified in myconfig_analysis.py)

You need tensorflow for evaluation. Run by:

CUDA_VISIBLE_DEVICES=0 python3 AnalyzeVideos.py

"""

####################################################
# Dependencies
####################################################

import os
import sys

# Add some folders to the python path
path_to_this_script = os.path.abspath(__file__)
path_to_delectable_folder = os.path.dirname(path_to_this_script)
dlc_folder_path = os.path.join(path_to_delectable_folder, "dlc")
sys.path.append(os.path.join(dlc_folder_path, "pose-tensorflow"))
sys.path.append(os.path.join(dlc_folder_path, "Generating_a_Training_Set"))
#sys.path.append(model_folder_path)

import dlct

#from myconfig_analysis import cropping, Task, date, \
#    trainingsFraction, resnet, snapshotindex, shuffle,x1, x2, y1, y2, videotype, storedata_as_csv

# Deep-cut dependencies
from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input

# Dependencies for video:
import pickle
# import matplotlib.pyplot as plt
import imageio
#imageio.plugins.ffmpeg.download()
from skimage.util import img_as_ubyte
from moviepy.editor import VideoFileClip
import skimage
import skimage.color
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

def getpose(image, cfg, outputs, outall=False):
    ''' Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose

def output_file_path_from_input_paths(model_folder_path, video_file_path, output_folder_path) :
    model_folder_name = dlct.file_name_without_extension_from_path(model_folder_path)
    video_file_folder_path = os.path.split(video_file_path)[0]
    video_file_name_without_extension = dlct.file_name_without_extension_from_path(video_file_path)
    output_file_name = video_file_name_without_extension + '-' + model_folder_name + '.h5'
    return os.path.join(output_folder_path, output_file_name)


# Get the command-line args
model_folder_path = os.path.abspath(sys.argv[1])
video_file_path = os.path.abspath(sys.argv[2])
if len(sys.argv) >= 4:
    output_file_path = os.path.abspath(sys.argv[3])
else:
    output_file_path = output_file_path_from_input_paths(model_folder_path, video_file_path, os.getcwd())

# Load the configuration file
configuration_file_name = 'myconfig.py'
configuration_file_path = os.path.join(model_folder_path, configuration_file_name)
# For backwards-compatibility:
if not os.path.exists(configuration_file_path):
    configuration_file_name = 'myconfig_analysis.py'
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
# Do some things to accomodate myconfig_analysis.py files
try:
    trainingFractionList = configuration['TrainingFraction']
except KeyError:
    trainingFractionList = [ configuration['trainingsFraction'] ]
try:
    shuffleList = configuration['Shuffles']
except KeyError:
    shuffleList = [1]

# These things are in myconfig_analysis.py in raw DeepLabCut
trainingFraction = trainingFractionList[0]
shuffle = shuffleList[0]
storedata_as_csv = True


####################################################
# Loading data, and defining model folder
####################################################

#basefolder = os.path.join('..','pose-tensorflow','models')
#basefolder = os.path.join(network_file_path, 'network')
modelfolder = os.path.join(model_folder_path,
                           Task + str(date) + '-trainset' + str(int(trainingFraction * 100)) + 'shuffle' + str(shuffle))

cfg = load_config(os.path.join(modelfolder , 'test' ,"pose_cfg.yaml"))

##################################################
# Load and setup CNN part detector
##################################################

# Check which snapshots are available and sort them by # iterations
Snapshots = np.array([
    fn.split('.')[0]
    for fn in os.listdir(os.path.join(modelfolder , 'train'))
    if "index" in fn
])
increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
Snapshots = Snapshots[increasing_indices]

print(modelfolder)
print(Snapshots)

##################################################
# Compute predictions over images
##################################################

# Check if data already was generated:
cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])

# Name for scorer:
trainingsiterations = (cfg['init_weights'].split('/')[-1]).split('-')[-1]

# Name for scorer:
scorer = 'DeepCut' + "_resnet" + str(resnet) + "_" + Task + str(
    date) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations)


cfg['init_weights'] = os.path.join(modelfolder , 'train', Snapshots[snapshotindex])
sess, inputs, outputs = predict.setup_pose_prediction(cfg)
pdindex = pd.MultiIndex.from_product(
    [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
    names=['scorer', 'bodyparts', 'coords'])

##################################################
# Datafolder
##################################################

# videofolder='../videos/' #where your folder with videos is.
frame_buffer = 10

#os.chdir(videofolder)
#videos = np.sort([fn for fn in os.listdir(os.curdir) if (videotype in fn)])
#print("Starting ", video_file_path)
#for video in videos:
video = video_file_path
#dataname = video.split('.')[0] + scorer + '.h5'

print("Loading ", video)
clip = VideoFileClip(video)
ny, nx = clip.size  # dimensions of frame (height, width)
fps = clip.fps
#nframes = np.sum(1 for j in clip.iter_frames()) #this is slow (but accurate)
nframes_approx = int(np.ceil(clip.duration * clip.fps) + frame_buffer)
# this will overestimage number of frames (see https://github.com/AlexEMG/DeepLabCut/issues/9) This is especially a problem
# for high frame rates and long durations due to rounding errors (as Rich Warren found). Later we crop the result (line 187)

if cropping:
    clip = clip.crop(
        y1=y1, y2=y2, x1=x1, x2=x2)  # one might want to adjust

print("Duration of video [s]: ", clip.duration, ", recorded with ", fps,
      "fps!")
print("Overall # of frames: ", nframes_approx,"with cropped frame dimensions: ", clip.size)

start = time.time()
PredicteData = np.zeros((nframes_approx, 3 * len(cfg['all_joints_names'])))
clip.reader.initialize()
print("Starting to extract posture")
highest_index_with_valid_frame = -1
#for image in clip.iter_frames():
for index in tqdm(range(nframes_approx)):
    image = img_as_ubyte(clip.reader.read_frame())
    # Thanks to Rick Warren for the  following snipplet:
    # if close to end of video, start checking whether two adjacent frames are identical
    # this should only happen when moviepy has reached the final frame
    # if two adjacent frames are identical, terminate the loop    
    if index==int(nframes_approx-frame_buffer*2):
        last_image = image
    elif index>int(nframes_approx-frame_buffer*2):
        if (image==last_image).all():
            #nframes = index
            #print("Detected frames: ", nframes)
            break
        else:
            last_image = image
    highest_index_with_valid_frame = index 
    pose = getpose(image, cfg, outputs)
    PredicteData[index, :] = pose.flatten()  # NOTE: thereby cfg['all_joints_names'] should be same order as bodyparts!
nframes = highest_index_with_valid_frame + 1 
print("Detected frames: ", nframes)

stop = time.time()

dictionary = {
    "start": start,
    "stop": stop,
    "run_duration": stop - start,
    "Scorer": scorer,
    "config file": cfg,
    "fps": fps,
    "frame_dimensions": (ny, nx),
    "nframes": nframes
}
metadata = {'data': dictionary}

print("Saving results...")
DataMachine = pd.DataFrame(PredicteData[:nframes,:], columns=pdindex, index=range(nframes)) #slice pose data to have same # as # of frames.
DataMachine.to_hdf(output_file_path, 'df_with_missing', format='table', mode='w')

if storedata_as_csv :
    csv_file_path = dlct.replace_extension(output_file_path, ".csv")
    DataMachine.to_csv(csv_file_path)

#with open(dataname.split('.')[0] + 'includingmetadata.pickle',
#          'wb') as f:
#    pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)
