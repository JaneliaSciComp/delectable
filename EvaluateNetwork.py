"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script evaluates a trained model at a particular state on the data set (images)
and stores the results in a pandas dataframe.

Script called from Step1_EvaluateModelonDataset.py

"""

####################################################
# Dependencies
####################################################

import sys
import os
#subfolder = os.getcwd().split('Evaluation-Tools')[0]
#sys.path.append(subfolder)

## add parent directory: (where nnet & config are!)
#sys.path.append(subfolder + "pose-tensorflow")
#sys.path.append(subfolder + "Generating_a_Training_Set")

#from myconfig import Task, date, Shuffles, scorer, TrainingFraction

# Deep-cut dependencies
from dlc.pose_tensorflow.config import load_config
from dlc.pose_tensorflow.nnet import predict
from dlc.pose_tensorflow.dataset.pose_dataset import data_to_input

# Dependencies for analysis
import pickle
import skimage
import numpy as np
import pandas as pd
from skimage import io
import skimage.color
#import dlc.Generating_a_Training_Set.auxiliaryfunctions
from tqdm import tqdm
#import dlct
import auxiliaryfunctions

# print("Starting evaluation") #, sys.argv)
# snapshotIndex=int(sys.argv[1])
# shuffleIndex=int(sys.argv[2])
# trainFractionIndex=int(sys.argv[3])
#
# shuffle=Shuffles[shuffleIndex]
# trainFraction=TrainingFraction[trainFractionIndex]
#
# working_directory_path = os.getcwd()
# parent_directory_path = os.path.dirname(working_directory_path)
# #basefolder = os.path.join('..', 'pose-tensorflow', 'models')
# models_folder_path = os.path.join(parent_directory_path, 'pose-tensorflow', 'models')


def dlc_evaluate_network(snapshotIndex, shuffleIndex, trainFractionIndex, models_folder_path, TrainingFraction, Shuffles, Task, date, scorer) :
    print("Starting evaluation")  # , sys.argv)

    shuffle = Shuffles[shuffleIndex]
    trainFraction = TrainingFraction[trainFractionIndex]

    unaugmented_folder_name = os.path.join('UnaugmentedDataSet_' + Task + date)

    pickle_file_name = ('Documentation_data-' + Task + '_' +
                        str(int(TrainingFraction[trainFractionIndex] * 100)) + 'shuffle' +
                        str(int(Shuffles[shuffleIndex])) + '.pickle')
    pickle_file_path = os.path.join(models_folder_path, unaugmented_folder_name, pickle_file_name)
    print("In EvaluateNetwork.py, pickle_file_path: %s" % pickle_file_path)

    # loading meta data / i.e. training & test files & labels
    try:
        with open(pickle_file_path, 'rb') as f:
            data, trainIndices, testIndices, __ignore__ = pickle.load(f)
    except FileNotFoundError:
        print('Caught the file not found error!')
        raise

    print('Now we are at the code after we caught the file not found error!')

    hdf_file_name = os.path.join(models_folder_path, unaugmented_folder_name, 'data-' + Task, 'CollectedData_' + scorer + '.h5')
    Data = pd.read_hdf(hdf_file_name, 'df_with_missing')

    #######################################################################
    # Load and setup CNN part detector as well as its configuration
    #######################################################################

    experimentname = Task + date + '-trainset' + str(int(trainFraction * 100)) + 'shuffle' + str(shuffle)
    cfg = load_config(os.path.join(models_folder_path, experimentname, 'test', "pose_cfg.yaml"))
    modelfolder = os.path.join(models_folder_path, experimentname)

    Snapshots = np.array([fn.split('.')[0]
        for fn in os.listdir(os.path.join(models_folder_path, experimentname, 'train'))
        if "index" in fn
    ])
    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    cfg['init_weights'] = os.path.join(modelfolder,'train',Snapshots[snapshotIndex])
    trainingsiterations = (
        cfg['init_weights'].split('/')[-1]).split('-')[-1]
    DLCscorer = 'DeepCut' + "_" + str(cfg["net_type"]) + "_" + str(
        int(trainFraction *
            100)) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations) + "forTask_" + Task

    print("Running ", DLCscorer, " with # of trainingiterations:", trainingsiterations)
    try:
        Data = pd.read_hdf(os.path.join("Results",DLCscorer + '.h5'),'df_with_missing')
        print("This net has already been evaluated!")
    except FileNotFoundError:
        # Specifying state of model (snapshot / training state)
        cfg['init_weights'] = os.path.join(modelfolder,'train',Snapshots[snapshotIndex])
        sess, inputs, outputs = predict.setup_pose_prediction(cfg)

        Numimages = len(Data.index)
        PredicteData = np.zeros((Numimages,3 * len(cfg['all_joints_names'])))
        Testset = np.zeros(Numimages)

        print("Analyzing data...")

        ##################################################
        # Compute predictions over images
        ##################################################

        for imageindex, imagename in tqdm(enumerate(Data.index)):
            image = io.imread(os.path.join(models_folder_path, unaugmented_folder_name, 'data-' + Task, imagename), mode='RGB')
            image = skimage.color.gray2rgb(image)
            image_batch = data_to_input(image)
            # Compute prediction with the CNN
            outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
            scmap, locref = predict.extract_cnn_output(outputs_np, cfg)

            # Extract maximum scoring location from the heatmap, assume 1 person
            pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
            PredicteData[imageindex, :] = pose.flatten(
            )  # NOTE: thereby     cfg_test['all_joints_names'] should be same order as bodyparts!

        index = pd.MultiIndex.from_product(
            [[DLCscorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
            names=['scorer', 'bodyparts', 'coords'])

        # Saving results:
        auxiliaryfunctions.attempttomakefolder("Results")

        DataMachine = pd.DataFrame(
            PredicteData, columns=index, index=Data.index.values)
        DataMachine.to_hdf(os.path.join("Results",DLCscorer + '.h5'),'df_with_missing',format='table',mode='w')
        print("Done and results stored for snapshot: ", Snapshots[snapshotIndex])



# #
# # main
# #
# if __name__ == '__main__' :
#     # If called this way, run on the model folder given as argument
#     working_directory_path = os.getcwd()
#     parent_directory_path = os.path.dirname(working_directory_path)
#     # basefolder = os.path.join('..', 'pose-tensorflow', 'models')
#     models_folder_path = os.path.join(parent_directory_path, 'pose-tensorflow', 'models')
#
#     snapshotIndex=int(sys.argv[1])
#     shuffleIndex=int(sys.argv[2])
#     trainFractionIndex=int(sys.argv[3])


