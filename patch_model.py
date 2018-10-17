#! /usr/bin/python3

import sys
import os
import tempfile
import shutil
import pwd


def get_username():
    return pwd.getpwuid(os.getuid())[0]


def is_empty(lst):
    return len(lst) == 0


def does_match_extension(file_name, target_extension):
    # target_extension should include the dot
    extension = os.path.splitext(file_name)[1]
    return extension == target_extension


def replace_extension(file_name, new_extension):
    # new_extension should include the dot
    base_name = os.path.splitext(file_name)[0]
    return base_name + new_extension


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
    names_of_matching_files = list(filter((lambda file_name: does_match_extension(file_name, output_file_extension)),
                                          names_of_files))
    return names_of_matching_files


class add_path:
    def __init__(self, path):
        self.original_sys_path = sys.path.copy()
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path = self.original_sys_path.copy()


def execfile(file_path, my_globals=None, my_locals=None):
    if my_globals is None:
        my_globals = {}
    my_globals.update({
        "__file__": file_path,
        "__name__": "__main__",
    })
    file_path = os.path.abspath(file_path)
    parent_folder_path = os.path.dirname(file_path)
    with add_path(parent_folder_path):
        with open(file_path, 'rt') as file:
            exec(compile(file.read(), file_path, 'exec'), my_globals, my_locals)


def load_configuration_file(file_path):
    my_globals = {}
    my_globals.update({
        "__file__": file_path,
        "__name__": "__main__",
    })
    my_locals = {}
    file_path = os.path.abspath(file_path)
    parent_folder_path = os.path.dirname(file_path)
    with add_path(parent_folder_path):
        with open(file_path, 'rt') as file:
            exec(compile(file.read(), file_path, 'exec'), my_globals, my_locals)
    return my_locals


def patch_model(targets_folder_path,
                original_model_folder_path,
                patched_model_folder_path):

    # Determine the absolute path to the "reference" DLC folder
    this_script_path = os.path.realpath(__file__)
    this_script_folder_path = os.path.dirname(this_script_path)
    template_dlc_root_folder_path = os.path.normpath(os.path.join(this_script_folder_path, 'dlc'))

    # Determine the absolute path to the parent temp folder that we can write to (e.g. /scratch/svobodalab)
    initial_working_folder_path = os.getcwd()
    print("username is %s" % get_username())
    scratch_folder_path = "/tmp"
    print("scratch_folder_path is %s" % scratch_folder_path)

    scratch_dlc_container_path_maybe = []  # want to keep track of this so we know whether or not to delete it
    # try:
    # Make a temporary folder to hold the temporary DLC folder
    # (want to have them for different instances running on same
    # node without collisions)
    scratch_dlc_container_path = tempfile.mkdtemp(prefix=scratch_folder_path + "/")
    scratch_dlc_container_path_maybe = [scratch_dlc_container_path]
    print("scratch_dlc_container_path is %s" % scratch_dlc_container_path)

    # Determine the absolute path to the temporary DLC folder
    scratch_dlc_root_folder_path = os.path.join(scratch_dlc_container_path, "dlc-working")
    print("scratch_dlc_root_folder_path is %s" % scratch_dlc_root_folder_path)

    # Copy the reference DLC folder to the scratch one
    shutil.copytree(template_dlc_root_folder_path, scratch_dlc_root_folder_path)

    # Load the configuration file
    configuration_file_name = 'myconfig.py'
    configuration_file_path = os.path.join(targets_folder_path, configuration_file_name)
    configuration = load_configuration_file(configuration_file_path)

    # Copy the configuration file into the scratch DLC folder
    scratch_configuration_file_path = os.path.join(scratch_dlc_root_folder_path, configuration_file_name)
    shutil.copyfile(configuration_file_path, scratch_configuration_file_path)

    # Copy the targets folder to the scratch DLC folder, in the right place
    task = configuration['Task']
    data_folder_name = 'data-' + task  # e.g. "data-licking-side"
    targets_folder_name = os.path.basename(targets_folder_path)
    scratch_targets_folder_path = os.path.join(scratch_dlc_root_folder_path,
                                               'Generating_a_Training_Set',
                                               data_folder_name,
                                               targets_folder_name)
    shutil.copytree(targets_folder_path, scratch_targets_folder_path)

    # Determine absolute path to the (scratch version of the) video-analysis script
    training_folder_path = os.path.join(scratch_dlc_root_folder_path, "Generating_a_Training_Set")
    make_data_frame_script_path = os.path.join(this_script_folder_path,
                                               "dlc",
                                               "Generating_a_Training_Set",
                                               "Step2_ConvertingLabels2DataFrame.py")
    print("training_folder_path: %s\n" % training_folder_path)

    # cd into the scratch analysis folder, run the scripts, cd back
    os.chdir(training_folder_path)
    # runpy.run_path(make_data_frame_script_path)
    # return_code = subprocess.call(['/usr/bin/python3', make_data_frame_script_path], shell=True)
    # if return_code != 0 :
    #    raise RuntimeError('There was a problem running, %s return code' % (make_data_frame_script_path, return_code))
    # execfile(make_data_frame_script_path)
    return_code = os.system('/usr/bin/python3 %s' % make_data_frame_script_path)
    if return_code != 0:
        raise RuntimeError(
            'There was a problem running, %s return code %d' % (make_data_frame_script_path, return_code))
    # import Step2_ConvertingLabels2DataFrame  # this is the one in the same folder as this file, hopefully

    make_training_file_script_path = os.path.join(this_script_folder_path,
                                                  "dlc",
                                                  "Generating_a_Training_Set",
                                                  "Step4_GenerateTrainingFileFromLabelledData.py")
    print("make_training_file_script_path: %s\n" % make_training_file_script_path)
    # runpy.run_path(make_training_file_script_path)
    # return_code = subprocess.call(['/usr/bin/python3', make_training_file_script_path], shell=True)
    # if return_code != 0 :
    #    raise RuntimeError('There was a problem running %s, return code %d'
    #                       % (make_training_file_script_path, return_code))
    # execfile(make_training_file_script_path)
    return_code = os.system('/usr/bin/python3 %s' % make_training_file_script_path)
    if return_code != 0:
        raise RuntimeError(
            'There was a problem running %s, return code %d' % (make_training_file_script_path, return_code))
    os.chdir(initial_working_folder_path)

    # Create the patched model folder
    os.makedirs(patched_model_folder_path)

    # Copy the test and train folders into a properly named subfolder of the patched model folder
    date = configuration['date']
    training_set_folder_name = task + date + "-trainset95shuffle1"
    destination_training_set_folder_path = \
        os.path.join(patched_model_folder_path,
                     training_set_folder_name)
    os.makedirs(destination_training_set_folder_path)
    shutil.copytree(os.path.join(original_model_folder_path, 'train'),
                    os.path.join(destination_training_set_folder_path, 'train'))
    shutil.copytree(os.path.join(original_model_folder_path, 'test'),
                    os.path.join(destination_training_set_folder_path, 'test'))

    # Copy the unaugmented training set folder from the scratch area to the patched model
    scratch_tensorflow_models_path = os.path.join(scratch_dlc_root_folder_path,
                                                  "pose-tensorflow",
                                                  "models")
    date = configuration['date']
    unaugmented_data_set_folder_name = "UnaugmentedDataSet_" + task + date
    source_unaugmented_data_set_folder_path = \
        os.path.join(scratch_dlc_root_folder_path,
                     "Generating_a_Training_Set",
                     unaugmented_data_set_folder_name)
    destination_unaugmented_data_set_folder_path = \
        os.path.join(patched_model_folder_path,
                     unaugmented_data_set_folder_name)
    shutil.copytree(source_unaugmented_data_set_folder_path,
                    destination_unaugmented_data_set_folder_path)

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


#
# main
#

# Get the arguments
targets_folder_path = os.path.abspath(sys.argv[1])
original_model_folder_path = os.path.abspath(sys.argv[2])
patched_model_folder_path = os.path.abspath(sys.argv[3])

# Run the training
patch_model(targets_folder_path, original_model_folder_path, patched_model_folder_path)
