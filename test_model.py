import sys
import os
import tempfile
import shutil
import dlct
#import EvaluateModelOnDataset
#import ComputeTestError

from config import load_config
from nnet import predict
from dataset.pose_dataset import data_to_input
import pickle
import skimage
import numpy as np
import pandas as pd
from skimage import io
import skimage.color
from tqdm import tqdm
import auxiliaryfunctions

#import numpy as np
#import pandas as pd
#from config import load_config

#import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from skimage import io

import argparse

def evaluate_network(snapshotIndex,
                     shuffleIndex,
                     trainFractionIndex,
                     model_folder_path,
                     TrainingFraction,
                     Shuffles,
                     Task,
                     date,
                     scorer,
                     model_on_targets_folder_path,
                     do_overwrite_files) :
    print("Starting evaluation")  # , sys.argv)

    shuffle = Shuffles[shuffleIndex]
    trainFraction = TrainingFraction[trainFractionIndex]

    unaugmented_folder_name = os.path.join('UnaugmentedDataSet_' + Task + date)

    pickle_file_name = ('Documentation_data-' + Task + '_' +
                        str(int(TrainingFraction[trainFractionIndex] * 100)) + 'shuffle' +
                        str(int(Shuffles[shuffleIndex])) + '.pickle')
    pickle_file_path = os.path.join(model_folder_path, unaugmented_folder_name, pickle_file_name)
    print("In EvaluateNetwork.py, pickle_file_path: %s" % pickle_file_path)

    # loading meta data / i.e. training & test files & labels
    try:
        with open(pickle_file_path, 'rb') as f:
            data, trainIndices, testIndices, __ignore__ = pickle.load(f)
    except FileNotFoundError:
        print('Caught the file not found error!')
        raise

    print('Now we are at the code after we caught the file not found error!')

    hdf_file_name = os.path.join(model_folder_path, unaugmented_folder_name, 'data-' + Task, 'CollectedData_' + scorer + '.h5')
    Data = pd.read_hdf(hdf_file_name, 'df_with_missing')

    #######################################################################
    # Load and setup CNN part detector as well as its configuration
    #######################################################################

    trainset_shuffle_folder_name = Task + date + '-trainset' + str(int(trainFraction * 100)) + 'shuffle' + str(shuffle)
    pose_cfg = load_config(os.path.join(model_folder_path, trainset_shuffle_folder_name, 'test', "pose_cfg.yaml"))
    trainset_shuffle_folder_path = os.path.join(model_folder_path, trainset_shuffle_folder_name)

    Snapshots = np.array([fn.split('.')[0]
        for fn in os.listdir(os.path.join(model_folder_path, trainset_shuffle_folder_name, 'train'))
        if "index" in fn
    ])
    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    pose_cfg['init_weights'] = os.path.join(trainset_shuffle_folder_path,'train',Snapshots[snapshotIndex])
    trainingsiterations = (
        pose_cfg['init_weights'].split('/')[-1]).split('-')[-1]
    DLCscorer = 'DeepCut' + "_" + str(pose_cfg["net_type"]) + "_" + str(
        int(trainFraction *
            100)) + 'shuffle' + str(shuffle) + '_' + str(trainingsiterations) + "forTask_" + Task

    print("Running ", DLCscorer, " with # of trainingiterations:", trainingsiterations)
    # Specifying state of model (snapshot / training state)
    pose_cfg['init_weights'] = os.path.join(trainset_shuffle_folder_path,'train',Snapshots[snapshotIndex])
    sess, inputs, outputs = predict.setup_pose_prediction(pose_cfg)

    Numimages = len(Data.index)
    PredicteData = np.zeros((Numimages,3 * len(pose_cfg['all_joints_names'])))
    Testset = np.zeros(Numimages)

    print("Analyzing data...")

    ##################################################
    # Compute predictions over images
    ##################################################

    for imageindex, imagename in tqdm(enumerate(Data.index)):
        image = io.imread(os.path.join(model_folder_path, unaugmented_folder_name, 'data-' + Task, imagename), mode='RGB')
        image = skimage.color.gray2rgb(image)
        image_batch = data_to_input(image)
        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref = predict.extract_cnn_output(outputs_np, pose_cfg)

        # Extract maximum scoring location from the heatmap, assume 1 person
        pose = predict.argmax_pose_predict(scmap, locref, pose_cfg.stride)
        PredicteData[imageindex, :] = pose.flatten(
        )  # NOTE: thereby     cfg_test['all_joints_names'] should be same order as bodyparts!

    index = pd.MultiIndex.from_product(
        [[DLCscorer], pose_cfg['all_joints_names'], ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

    # Saving results:
    if not os.path.exists(model_on_targets_folder_path) :
        os.mkdir(model_on_targets_folder_path)

    DataMachine = pd.DataFrame(
        PredicteData, columns=index, index=Data.index.values)
    hdf_file_name = DLCscorer + '.h5'
    hdf_file_path = os.path.join(model_on_targets_folder_path, hdf_file_name)
    DataMachine.to_hdf(hdf_file_path,'df_with_missing',format='table',mode='w')
    print("Done and results stored for snapshot: ", Snapshots[snapshotIndex])


def evaluate_model_on_dataset(model_folder_path, configuration, do_overwrite_files, model_on_targets_folder_path):
    # Load the configuration file
    #configuration_file_name = 'myconfig.py'
    #configuration_file_path = os.path.join(model_folder_path, configuration_file_name)
    #configuration = dlct.load_configuration_file(configuration_file_path)
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

    cfg = load_config(os.path.join(trainset_shuffle_folder_path, 'test', "pose_cfg.yaml"))
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
        # if not analyzed, then call auxiliary script to analyze the network
        #return_code = subprocess.call(['python3','EvaluateNetwork.py',str(snapIndex),str(shuffleIndex),str(trainFractionIndex)])
        #if return_code != 0:
        #    raise RuntimeError(
        #        'There was a problem running EvaluateNetwork.py, return code was %d' % return_code)
        evaluate_network(snapIndex,
                         shuffleIndex,
                         trainFractionIndex,
                         model_folder_path,
                         TrainingFraction,
                         Shuffles,
                         Task,
                         date,
                         configuration['scorer'],
                         model_on_targets_folder_path,
                         do_overwrite_files)


def MakeLabeledImage(DataCombined,imagenr,imagefilename,Scorers,bodyparts,colors, pcutoff) :
    '''Creating a labeled image with the original human labels, as well as the DeepLabCut's!'''
    labels=['+','.','x']
    scaling=1
    alphavalue=.5
    dotsize=15
    plt.axis('off')
    im=io.imread(os.path.join(imagefilename,DataCombined.index[imagenr]))
    if np.ndim(im)>2:
        h,w,numcolors=np.shape(im)
    else: #grayscale
        h,w=np.shape(im)
    plt.figure(frameon=False,figsize=(w*1./100*scaling,h*1./100*scaling))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.imshow(im,'gray')
    for scorerindex,loopscorer in enumerate(Scorers):
       for bpindex,bp in enumerate(bodyparts):
           if np.isfinite(DataCombined[loopscorer][bp]['y'][imagenr]+DataCombined[loopscorer][bp]['x'][imagenr]):
                y,x=int(DataCombined[loopscorer][bp]['y'][imagenr]), int(DataCombined[loopscorer][bp]['x'][imagenr])
                if 'DeepCut' in loopscorer:
                    p=DataCombined[loopscorer][bp]['likelihood'][imagenr]
                    if p>pcutoff:
                        plt.plot(x,y,labels[1],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
                    else:
                        plt.plot(x,y,labels[2],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
                else: #by exclusion this is the human labeler (I hope nobody has DeepCut in his name...)
                        plt.plot(x,y,labels[0],ms=dotsize,alpha=alphavalue,color=colors(int(bpindex)))
    plt.xlim(0,w)
    plt.ylim(0,h)
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.gca().invert_yaxis()
    return 0


def pairwisedistances(DataMachine, DataCombined,scorer1,scorer2,pcutoff=-1,bodyparts=None):
    ''' Calculates the pairwise Euclidean distance metric '''
    mask=DataMachine[scorer2].xs('likelihood',level=1,axis=1)>=pcutoff
    if bodyparts==None:
            Pointwisesquareddistance=(DataCombined[scorer1]-DataCombined[scorer2])**2
            RMSE=np.sqrt(Pointwisesquareddistance.xs('x',level=1,axis=1)+Pointwisesquareddistance.xs('y',level=1,axis=1)) #Euclidean distance (proportional to RMSE)
            return RMSE,RMSE[mask]
    else:
            Pointwisesquareddistance=(DataCombined[scorer1][bodyparts]-DataCombined[scorer2][bodyparts])**2
            RMSE=np.sqrt(Pointwisesquareddistance.xs('x',level=1,axis=1)+Pointwisesquareddistance.xs('y',level=1,axis=1)) #Euclidean distance (proportional to RMSE)
            return RMSE,RMSE[mask]


def compute_test_error(model_folder_path,
                       model_on_targets_folder_path,
                       task,
                       date,
                       scorer,
                       Shuffles,
                       TrainingFraction,
                       snapshotindex,
                       pcutoff,
                       plotting,
                       do_overwrite_outputs,
                       colormap):
    unaugmented_folder_name = 'UnaugmentedDataSet_' + task + date
    unaugmented_folder_path = os.path.join(model_folder_path, unaugmented_folder_name)

    data = pd.read_hdf(os.path.join(unaugmented_folder_path, 'data-' + task, 'CollectedData_' + scorer + '.h5'), 'df_with_missing')

    ####################################################
    # Models vs. benchmark for varying training state
    ####################################################

    # only specific parts can also be compared (not all!) (in that case change which bodyparts by providing a list below)
    comparisonbodyparts = list(np.unique(data.columns.get_level_values(1)))
    colors = None
    if plotting==True:
        #colors = dlct.get_cmap(len(comparisonbodyparts))
        colors = dlct.get_repeated_cmap(colormap, len(comparisonbodyparts))

    for trainFraction in TrainingFraction:
        for shuffle in Shuffles:

            fns = [file for file in os.listdir(model_on_targets_folder_path)
                   if "forTask_" + str(task) in file and "shuffle" + str(shuffle) in
                   file and "_" + str(int(trainFraction * 100)) in file]

            metadata_file_name = os.path.join(unaugmented_folder_path, "Documentation_" + "data-" + task + "_" + str(
                int(trainFraction * 100)) + "shuffle" + str(shuffle) + ".pickle")
            with open(metadata_file_name, 'rb') as f:
                [
                    trainingdata_details, trainIndexes, testIndexes,
                    testFraction_data
                ] = pickle.load(f)

            # extract training iterations:
            TrainingIterations = [(int(fns[j].split("forTask")[0].split('_')[-1]), j) for j in range(len(fns))]
            TrainingIterations.sort(key=lambda tup: tup[0])  # sort according to increasing # training steps!

            print("Assessing accuracy of shuffle # ", shuffle, " with ", int(trainFraction * 100),
                  " % training fraction.")
            print("Found the following training snapshots: ", TrainingIterations)
            print("You can choose among those for analyis of train/test performance.")

            if snapshotindex == -1:
                snapindices = [TrainingIterations[-1]]
            elif snapshotindex == "all":
                snapindices = TrainingIterations
            elif snapshotindex < len(TrainingIterations):
                snapindices = [TrainingIterations[snapshotindex]]
            else:
                print(
                    "Invalid choice, only -1 (last), all (as string), or index corresponding to one of the listed training snapshots can be analyzed.")
                print("Others might not have been evaluated!")
                snapindices = []

            for trainingiterations, index in snapindices:
                DataMachine = pd.read_hdf(os.path.join(model_on_targets_folder_path, fns[index]), 'df_with_missing')
                DataCombined = pd.concat([data.T, DataMachine.T], axis=0).T
                scorer_machine = DataMachine.columns.get_level_values(0)[0]
                RMSE, RMSEpcutoff = pairwisedistances(DataMachine, DataCombined, scorer, scorer_machine, pcutoff,
                                                      comparisonbodyparts)
                testerror = np.nanmean(RMSE.iloc[testIndexes].values.flatten())
                trainerror = np.nanmean(RMSE.iloc[trainIndexes].values.flatten())
                testerrorpcutoff = np.nanmean(RMSEpcutoff.iloc[testIndexes].values.flatten())
                trainerrorpcutoff = np.nanmean(RMSEpcutoff.iloc[trainIndexes].values.flatten())
                print("Results for", trainingiterations, "training iterations:", int(100 * trainFraction), shuffle,
                      "train error:", np.round(trainerror, 2), "pixels. Test error:", np.round(testerror, 2),
                      " pixels.")
                print("With pcutoff of", pcutoff, " train error:", np.round(trainerrorpcutoff, 2),
                      "pixels. Test error:", np.round(testerrorpcutoff, 2), "pixels")
                print(
                    "Thereby, the errors are given by the average distances between the labels by DLC and the scorer.")

                if plotting == True:
                    model_folder_name = dlct.file_name_without_extension_from_path(model_folder_path)
                    images_folder_name = model_folder_name + "-target-images-with-predictions"
                    #images_folder_name = os.path.join('LabeledImages_' + scorer_machine)
                    if os.path.exists(images_folder_name) :
                        if do_overwrite_outputs :
                            shutil.rmtree(images_folder_name)
                        else :
                            raise RuntimeError('Images folder %s already exists---not overwriting' % images_folder_name)
                    os.mkdir(images_folder_name)
                    numFrames = np.size(DataCombined.index)
                    for ind in np.arange(numFrames):
                        fn = DataCombined.index[ind]

                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        MakeLabeledImage(DataCombined,
                                         ind,
                                         os.path.join(unaugmented_folder_path, 'data-' + task),
                                         [scorer, scorer_machine],
                                         comparisonbodyparts,
                                         colors,
                                         pcutoff)
                        if ind in trainIndexes:
                            plt.savefig(os.path.join(images_folder_name,
                                                     'TrainingImg' + str(ind) + '_' + fn.split('/')[0] + '_' +
                                                     fn.split('/')[1]))
                        else:
                            plt.savefig(os.path.join(images_folder_name,
                                                     'TestImg' + str(ind) + '_' + fn.split('/')[0] + '_' + fn.split('/')[
                                                         1]))
                        plt.clf()
                        plt.close("all")


def find_files_matching_extension(output_folder_path, output_file_extension):
    # get a list of all files and folders in the output_folder_path
    try:
        names_of_files_and_subfolders = os.listdir(output_folder_path)
    except FileNotFoundError:
        # if we can't list the dir, warn but continue
        raise RuntimeError("Warning: Folder %s doesn't seem to exist" % output_folder_path)
    except PermissionError:
        # if we can't list the dir, warn but continue
        raise RuntimeError("Warning: can't list contents of folder %s due to permissions error" % output_folder_path)

    names_of_files = list(filter((lambda item_name: os.path.isfile(os.path.join(output_folder_path, item_name))),
                                 names_of_files_and_subfolders))
    names_of_matching_files = list(filter((lambda file_name: dlct.does_match_extension(file_name, output_file_extension)),
                                          names_of_files))
    return names_of_matching_files


def test_model(targets_folder_path, model_folder_path, do_produce_images, do_overwrite_outputs):
    # Load the configuration file
    configuration_file_name = 'myconfig.py'
    configuration_file_path = os.path.join(targets_folder_path, configuration_file_name)
    configuration = dlct.load_configuration_file(configuration_file_path)

    # Construct the name of the folder where we'll put the evaluation results
    model_folder_name = os.path.basename(model_folder_path)
    model_on_targets_folder_name = model_folder_name + "-on-targets"
    model_on_targets_folder_path = os.path.abspath(model_on_targets_folder_name)

    # See if the model-on-targets folder exists.  If it does, error or overwrite
    if os.path.exists(model_on_targets_folder_path) :
        if do_overwrite_outputs :
            shutil.rmtree(model_on_targets_folder_path)
        else :
            raise RuntimeError('Model-on-targets folder %s already exists---not overwriting or modifying' %
                               model_on_targets_folder_name)

    # Run the first script
    evaluate_model_on_dataset(model_folder_path, configuration, do_overwrite_outputs, model_on_targets_folder_path)

    # Run the second script
    task = configuration['Task']
    date = configuration['date']
    scorer = configuration['scorer']
    Shuffles = configuration['Shuffles']
    TrainingFraction = configuration['TrainingFraction']
    snapshotindex = configuration['snapshotindex']
    pcutoff = configuration['pcutoff']
    colormap = configuration['colormap']
    compute_test_error(model_folder_path,
                       model_on_targets_folder_path,
                       task,
                       date,
                       scorer,
                       Shuffles,
                       TrainingFraction,
                       snapshotindex,
                       pcutoff,
                       do_produce_images,
                       do_overwrite_outputs,
                       colormap)



#
# main
#

if __name__ == '__main__' :
    # Get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--force", help="overwrite output files",
                        action="store_true")
    parser.add_argument("-i", "--images", help="produce output images",
                        action="store_true")
    parser.add_argument("targets_folder_name", help="folder of training targets")
    parser.add_argument("model_folder_name", help="folder containing a model")
    args = parser.parse_args()

    targets_folder_path = os.path.abspath(args.targets_folder_name)
    model_folder_path = os.path.abspath(args.model_folder_name)
    do_produce_images = args.images
    do_overwrite_outputs = args.force

    #targets_folder_path = os.path.abspath(sys.argv[1])
    #model_folder_path = os.path.abspath(sys.argv[2])

    # Run the training
    test_model(targets_folder_path, model_folder_path, do_produce_images, do_overwrite_outputs)
