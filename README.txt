Delectable

This project provides tools to apply DeepLabCut to videos, with a
workflow that is easier in some ways than that provided by 'raw'
DeepLabCut.

This project is based on the DeepLabCut project:

https://github.com/AlexEMG/DeepLabCut

All the code under the dlc folder is very close to the original
DeepLabCut code, and therefore carries its own copyright and license
(GPLv3).


Warning
-------

This code works for me.  It probably won't work for you
out-of-the-box.  Sorry about that.  Feel free to e-mail me with
questions.


Usage
-----

Once everything is set up right, you should be able to do this in
Bash:

train_model <targets-folder-name> <model-folder-name>

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
docs for DeepLabCut.)  Each .csv contains a single fiducial point for
each training frame.  These should be in the same format that
DeepLabCut normally expects (you can make them with Fiji, for
instance.)  There should be one row per training frame, and the nth
row should correspond to the nth training frame, if the training
frames were sorted using a *natural* sort by file name.  See here for
what that means:

https://en.wikipedia.org/wiki/Natural_sort_order

After running for a long time (order 24 hrs), <model-folder-name> will
contain a DeepLabCut model.  This will contain one folder with a name
of the form <task><date>-trainset95shuffle1, and one folder with
a name of the form UnaugmentedDataSet_<task><date>.  These are the
same as the ones that would be produced by raw DeepLabCut in the
pose-tensorflow/models folder.  They contain a complete description of
your model, sufficient to apply it to new models.

Once you have a trained model, you can do:

test_model <targets-folder-name> <model-folder-name>

This will run your model on training frames that were ommitted during
training, and will print the training and test errors (in pixels) to
stdout.

There are two additional bash scripts:

train_model_on_cluster
test_model_on_cluster

These are like the scripts described above, but they will be submitted
as jobs on an LSF cluster, inside a singularity container.  For this
to work, of course, they have to be run from a submit host for an LSF
cluster.  You also have to build the singularity container.  We provide
the recipe for this (dlc.def), and how to build the container is
described in

how-to-build-singularity-image.txt

ALT
2018-10-29

