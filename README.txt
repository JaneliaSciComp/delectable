Delectable

This project provides tools to apply DeepLabCut (version 1) to videos,
with a workflow that is easier in some ways than that provided by
'raw' DeepLabCut.  It assumes you are at Janelia, and logged in to a
one of the login nodes of the Janelia cluster.

This project is based on the DeepLabCut project:

https://github.com/AlexEMG/DeepLabCut

All the code under the dlc folder is very close to the original
DeepLabCut code, and therefore carries its own copyright and license
(GPLv3).


Installation
------------

Add the folder /groups/svoboda/home/svobodalab/delectable to your
PATH.  E.g. add the following line to your ~/.profile file: 

PATH=$PATH:/groups/svoboda/home/svobodalab/delectable

then log out and log back in again.  (Nerds: Yes, I know there are
other ways to accomplish this.  This one is simple.)


Usage
-----

ssh into to login1.int.janelia.org or login2.int.janelia.org.  If
you're on MacOS, do this from the Terminal.  If Windows, recent
versions should ship with ssh.  Failing that, use MobaXTerm or the NX
client.

Once you're logged in, you should be able to do this in
the terminal:

train_model_on_cluster <targets-folder-name> <model-folder-name>

This takes as input <targets-folder-name>, and produces output in
<model-folder-name>.  It will probably overwrite a pre-exisiting
model-folder-name, so be careful.  The contents of targets-folder-name
should be something like:

myconfig.py
jaw.csv
tongue.csv
nose.csv
<a bunch of training frames, each a .png>

The myconfig.py is a DeepLabCut-style myconfig.py file.  (Refer to the
docs for DeepLabCut 1.)  Each .csv contains a single fiducial point
(aka a body part marker) for each training frame.  These should be in
the same format that DeepLabCut normally expects (you can make them
with Fiji, for instance.)  There should be one row per training frame,
and the nth row should correspond to the nth training frame, if the
training frames were sorted using a *natural* sort by file name.  See
here for what that means:

https://en.wikipedia.org/wiki/Natural_sort_order

After running for a long time (order 24 hrs), <model-folder-name> will
contain a DeepLabCut model.  (You can check if your job is running
with bjobs and all the usual cluster tools.)  This will contain one
folder with a name of the form <task><date>-trainset95shuffle1, one
folder with a name of the form UnaugmentedDataSet_<task><date>, and a
copy of your myconfig.py file from the targets folder.  These are the
same as the ones that would be produced by raw DeepLabCut in the
pose-tensorflow/models folder.  They contain a complete description of
your model, sufficient to apply it to new models.

Once you have a trained model, you can do:

test_model_on_cluster <targets folder name> <model folder name>

This will run your model on training frames that were ommitted during
training, and will print the training and test errors (in pixels) to
stderr.

To apply your trained model to novel videos, do:

apply_model_on_cluster <model folder name> <video file name> <h5 file_name>

This will produce an HDF5 file (typically with a name ending in .h5)
containing the model predictions for the video.  Depending on your
myconfig.py, it may also produce a .csv file with the same
information.

To visualize the model predictions superimposed on the video, do:

make_labeled_video_on_cluster <model folder name> <video file name> <h5 file_name> <output video file name>

This will produce a video file (with extension .mp4, typically)
showing the labels in <h5 file name> superimposed on <video file
name>.

We also provide a script to compress videos in a way that provides a
good tradeoff between compression and quality:

compress_videos_on_cluster <input folder name> <output folder name>

Will look for video files in <input folder name> and will compress
each one, outputing to a parallel location in <output folder name>.
It will use HEVC (H.265) compression, using a CRF of 35 at 1 FPS.  All
output files will be (nominally) 1 FPS, but with the same number of
frames as the original.  The output files typically acheive 100x
compression, with minimal (but nonzero) loss of quality.

ALT
2018-12-03


