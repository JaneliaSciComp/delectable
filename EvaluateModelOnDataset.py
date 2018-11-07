"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

This script evaluates a trained model at a particular state on the data set (images)
and stores the results in a pandas dataframe.

You need tensorflow for evaluation. Run by:

CUDA_VISIBLE_DEVICES=0 python3 Step1_EvaluateModelonDataset.py

"""

####################################################
# Dependencies
####################################################

import sys
import os
import subprocess

import dlct
#subfolder = os.getcwd().split('Evaluation-Tools')[0]
#sys.path.append(subfolder)

# add parent directory: (where nnet & config are!)
#sys.path.append(subfolder + "pose-tensorflow")
#sys.path.append(subfolder + "Generating_a_Training_Set")

#from myconfig import Task, date, Shuffles, TrainingFraction, snapshotindex
import numpy as np
import pandas as pd
# Deep-cut dependencies
from dlc.pose_tensorflow.config import load_config

import EvaluateNetwork

def dlc_evaluate_model_on_dataset(model_folder_path):
    # Load the configuration file
    configuration_file_name = 'myconfig.py'
    configuration_file_path = os.path.join(model_folder_path, configuration_file_name)
    configuration = dlct.load_configuration_file(configuration_file_path)
    Task = configuration['Task']
    date = configuration['date']
    Shuffles = configuration['Shuffles']
    TrainingFraction = configuration['TrainingFraction']
    snapshotindex = configuration['snapshotindex']

    shuffleIndex = 0
    shuffle = Shuffles[shuffleIndex]
    trainFractionIndex = 0
    trainFraction = TrainingFraction[trainFractionIndex]

    trainset_shuffle_folder_name = Task + date + '-trainset' + str(int(trainFraction * 100)) + 'shuffle' + str(shuffle)
    trainset_shuffle_folder_path = os.path.join(model_folder_path, trainset_shuffle_folder_name)

    cfg = load_config(os.path.join(trainset_shuffle_folder_path , 'test' ,"pose_cfg.yaml"))
    # Check which snap shots are available and sort them by # iterations
    Snapshots = np.array([
        fn.split('.')[0]
        for fn in os.listdir(os.path.join(trainset_shuffle_folder_path , 'train'))
        if "index" in fn])

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    if snapshotindex == -1:
        snapindices = [-1]
    elif snapshotindex == "all":
        snapindices = range(len(Snapshots))
    elif snapshotindex<len(Snapshots):
        snapindices=[snapshotindex]
    else:
        raise RuntimeError("Invalid choice, only -1 (last), any integer up to last, or all (as string)!")

    for snapIndex in snapindices:
        cfg['init_weights'] = os.path.join(trainset_shuffle_folder_path,'train',Snapshots[snapIndex])
        trainingsiterations = (cfg['init_weights'].split('/')[-1]).split('-')[-1]
        scorer = 'DeepCut' + "_" + str(cfg["net_type"]) + "_" + str(
            int(trainFraction *
                100)) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations) + "forTask_" + Task

        print("Running ", scorer, " with # of trainingiterations:", trainingsiterations)
        results_hdf_file_name = os.path.join("Results", scorer + '.h5')
        if os.path.isfile(results_hdf_file_name) :
            #Data = pd.read_hdf(os.path.join("Results", scorer + '.h5'), 'df_with_missing')
            print("This net has already been evaluated!")
        else :
            # if not analyzed, then call auxiliary script to analyze the network
            #return_code = subprocess.call(['python3','EvaluateNetwork.py',str(snapIndex),str(shuffleIndex),str(trainFractionIndex)])
            #if return_code != 0:
            #    raise RuntimeError(
            #        'There was a problem running EvaluateNetwork.py, return code was %d' % return_code)
            EvaluateNetwork.dlc_evaluate_network(snapIndex, shuffleIndex, trainFractionIndex, model_folder_path,
                                                 TrainingFraction,
                                                 Shuffles, Task, date, configuration['scorer'])



#
# main
#
if __name__ == '__main__' :
    # If called this way, run on the model folder given as argument
    model_folder_path = os.path.abspath(sys.argv[1])
    dlc_evaluate_model_on_dataset(model_folder_path)

