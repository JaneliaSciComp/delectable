#! /usr/bin/python3

import sys
import os
import tempfile
import shutil
#import runpy
import subprocess
import pwd
import pathlib

def get_username():
    return pwd.getpwuid( os.getuid() )[ 0 ]

def is_empty(list) :
    return len(list)==0

def does_match_extension(file_name, target_extension) :
    # target_extension should include the dot
    extension = os.path.splitext(file_name)[1]
    return (extension == target_extension)

def replace_extension(file_name, new_extension) :
    # new_extension should include the dot
    base_name = os.path.splitext(file_name)[0]
    return base_name + new_extension

def find_files_matching_extension(output_folder_path, output_file_extension) :
    # get a list of all files and folders in the output_folder_path
    try:
        names_of_files_and_subfolders = os.listdir(output_folder_path)
    except FileNotFoundError :
        # if we can't list the dir, warn but continue
        raise RuntimeError("Warning: Folder %s doesn't seem to exist" % output_folder_path)   
    except PermissionError :
        # if we can't list the dir, warn but continue
        raise RuntimeError("Warning: can't list contents of folder %s due to permissions error" % output_folder_path)
    
    names_of_files = list(filter((lambda item_name: os.path.isfile(os.path.join(output_folder_path, item_name))) , 
                                 names_of_files_and_subfolders))
    names_of_matching_files = list(filter((lambda file_name: does_match_extension(file_name, output_file_extension)) , 
                                          names_of_files))
    return names_of_matching_files
# end of function

class add_path():
    def __init__(self, path):
        self.original_sys_path = sys.path.copy()

    def __enter__(self):
        sys.path.insert(0, self.original_sys_path)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path = self.original_sys_path.copy()
# end of class


def execfile(file_path, globals=None, locals=None):
    if globals is None:
        globals = {}
    globals.update({
        "__file__": file_path,
        "__name__": "__main__",
    })
    file_path = os.path.abspath(file_path)
    parent_folder_path = os.path.dirname(file_path)
    with add_path(parent_folder_path) :
        with open(file_path, 'rt') as file:
            exec(compile(file.read(), file_path, 'exec'), globals, locals)
# end of function


def train(targets_folder_path, 
          network_folder_path):

    # Determine the absolute path to the "reference" DLC folder
    this_script_path = os.path.realpath(__file__)
    this_script_folder_path = os.path.dirname(this_script_path)
    template_dlc_root_folder_path = os.path.normpath(os.path.join(this_script_folder_path, "dlc-working-template"))

    # Determine the absolute path to the parent temp folder that we can write to (e.g. /scratch/svobodalab)
    initial_working_folder_path = os.getcwd()
    print("username is %s" % get_username()) 
    scratch_folder_path = "/tmp"
    print("scratch_folder_path is %s" % scratch_folder_path) 

    scratch_dlc_container_path_maybe = []  # want to keep track of this so we know whether or not to delete it
    #try:
    # Make a temporary folder to hold the temporary DLC folder
    # (want to have them for different instances running on same
    # node without collisions)
    scratch_dlc_container_path = tempfile.mkdtemp(prefix=scratch_folder_path+"/")
    scratch_dlc_container_path_maybe = [scratch_dlc_container_path]
    print("scratch_dlc_container_path is %s" % scratch_dlc_container_path)

    # Determine the absolute path to the temporary DLC folder
    scratch_dlc_root_folder_path = os.path.join(scratch_dlc_container_path, "dlc-working")
    print("scratch_dlc_root_folder_path is %s" % scratch_dlc_root_folder_path)

    # Copy the reference DLC folder to the scratch one
    shutil.copytree(template_dlc_root_folder_path, scratch_dlc_root_folder_path)

    # Copy the configuration file into the scratch DLC folder
    configuration_file_name = "myconfig.py"
    configuration_file_path = os.path.join(targets_folder_path, configuration_file_name)
    scratch_configuration_file_path = os.path.join(scratch_dlc_root_folder_path, configuration_file_name)
    shutil.copyfile(configuration_file_path, scratch_configuration_file_path)

    # Copy the targets folder to the scratch DLC folder, in the right place
    data_folder_name = "data-" + Task  # e.g. "data-licking-side"
    targets_folder_name = os.path.basename(targets_folder_path)
    scratch_targets_folder_path = os.path.join(scratch_dlc_root_folder_path, "Generating_a_Training_Set", data_folder_name, targets_folder_name)
    shutil.copytree(targets_folder_path, scratch_targets_folder_path)

    # Determine absolute path to the (scratch version of the) video-analysis script
    training_folder_path = os.path.join(scratch_dlc_root_folder_path, "Generating_a_Training_Set")
    make_data_frame_script_path = os.path.join(this_script_folder_path, "dlc", "Generating_a_Training_Set", "Step2_ConvertingLabels2DataFrame.py")
    print("training_folder_path: %s\n" % training_folder_path)

    # cd into the scratch analysis folder, run the scripts, cd back
    os.chdir(training_folder_path)
    #runpy.run_path(make_data_frame_script_path)
    #return_code = subprocess.call(['/usr/bin/python3', make_data_frame_script_path], shell=True)
    #if return_code != 0 :
    #    raise RuntimeError('There was a problem running, %s return code' % (make_data_frame_script_path, return_code))
    #execfile(make_data_frame_script_path)
    return_code = os.system('/usr/bin/python3 %s' %  make_data_frame_script_path)
    if return_code != 0 :
        raise RuntimeError('There was a problem running, %s return code %d' % (make_data_frame_script_path, return_code))
    #import Step2_ConvertingLabels2DataFrame  # this is the one in the same folder as this file, hopefully

    #make_training_file_script_path = os.path.join(training_folder_path, "Step4_GenerateTrainingFileFromLabelledData.py")
    make_training_file_script_path = os.path.join(this_script_folder_path, "dlc", "Generating_a_Training_Set", "Step4_GenerateTrainingFileFromLabelledData.py")
    print("make_training_file_script_path: %s\n" % make_training_file_script_path)
    #runpy.run_path(make_training_file_script_path)
    #return_code = subprocess.call(['/usr/bin/python3', make_training_file_script_path], shell=True)
    #if return_code != 0 :
    #    raise RuntimeError('There was a problem running %s, return code %d' % (make_training_file_script_path, return_code))
    #execfile(make_training_file_script_path)
    return_code = os.system('/usr/bin/python3 %s' %  make_training_file_script_path)
    if return_code != 0 :
        raise RuntimeError('There was a problem running %s, return code %d' % (make_training_file_script_path, return_code))
    os.chdir(initial_working_folder_path)

    # Copy the relevant folders over to the pose-tensorflow folder
    tensorflow_models_path = os.path.join(scratch_dlc_root_folder_path, "pose-tensorflow", "models")
    trainset_folder_name =  Task + date + "-trainset95shuffle1"
    source_trainset_folder_path = os.path.join(scratch_dlc_root_folder_path, "Generating_a_Training_Set", trainset_folder_name)
    dest_trainset_folder_path = os.path.join(tensorflow_models_path, trainset_folder_name)
    unaugmented_data_set_folder_name =  "UnaugmentedDataSet_" + Task + date
    unaugmented_data_set_folder_path = os.path.join(scratch_dlc_root_folder_path, "Generating_a_Training_Set", unaugmented_data_set_folder_name)
    dest_unaugmented_data_set_folder_path = os.path.join(tensorflow_models_path, unaugmented_data_set_folder_name)
    shutil.copytree(source_trainset_folder_path, dest_trainset_folder_path)
    shutil.copytree(unaugmented_data_set_folder_path, dest_unaugmented_data_set_folder_path)

    # cd into the (scratch) folder, run the training script, cd back
    folder_for_running_training_path = os.path.join(tensorflow_models_path, trainset_folder_name, "train")
    #training_script_path = os.path.join(scratch_dlc_root_folder_path, "pose-tensorflow", "train.py")
    training_script_path = os.path.join(this_script_folder_path, "dlc", "pose-tensorflow", "train.py")
    os.chdir(folder_for_running_training_path)
    #runpy.run_path(training_script_path)
    #return_code = subprocess.call(['/usr/bin/python3', training_script_path], shell=True)
    #if return_code != 0 :
    #    raise RuntimeError('There was a problem running %s, return code %d' % (training_script_path, return_code))
    #execfile(training_script_path)
    os.system('/usr/bin/python3 %s' % training_script_path)
    os.chdir(initial_working_folder_path)

    # Copy the scratch network folder output file location
    scratch_network_folder_path = dest_trainset_folder_path
    print("About to copy result to final location...")
    print("scratch_network_folder_path: %s" % scratch_network_folder_path)
    print("network_folder_path: %s" % network_folder_path)
    shutil.copytree(scratch_network_folder_path, network_folder_path)

    # Remove the scratch folder we created to hold the scratch DLC folder
    shutil.rmtree(scratch_dlc_container_path)

    # except Exception as e:
    #     # Try to clean up some before re-throwing
    #     print("Something went wrong!")
    #
    #     # Remove the scratch folder
    #     if not is_empty(scratch_dlc_container_path_maybe) :
    #         scratch_dlc_container_path = scratch_dlc_container_path_maybe[0]
    #         print("Leaving scratch folder in place at %s" % scratch_dlc_container_path)
    #
    #     # cd back to the initial folder
    #     os.chdir(initial_working_folder_path)
    #
    #     # Re-throw the exception
    #     raise e
# end of function


# main
targets_folder_path = os.path.abspath(sys.argv[1])
network_folder_path = os.path.abspath(sys.argv[2])

sys.path.append(targets_folder_path)
from myconfig import Task, date

train(targets_folder_path, network_folder_path)
