#! /usr/bin/python3

import sys
import os
import tempfile
import shutil
import platform
import dlct


def train_model(targets_folder_path,
                model_folder_path):
    # Set the python executable path (Use same executable as is running now)
    python_executable_path = sys.executable
    # if platform.system() == 'Windows':
    #     python_executable_path = 'C:/Users/taylora/AppData/Local/Programs/Python/Python36/python.exe'
    # else:
    #     python_executable_path = '/usr/bin/python3'

    # Determine the absolute path to the "reference" DLC folder
    this_script_path = os.path.realpath(__file__)
    delectable_folder_path = os.path.dirname(this_script_path)
    template_dlc_root_folder_path = os.path.normpath(os.path.join(delectable_folder_path, 'dlc'))

    # Determine the absolute path to the parent temp folder that we can write to (e.g. /scratch/svobodalab)
    initial_working_folder_path = os.getcwd()
    #print("username is %s" % dlct.get_username())
    generalized_slash_tmp_path = dlct.determine_scratch_folder_path()
    print("generalized_slash_tmp_path is %s" % generalized_slash_tmp_path)

    scratch_dlc_container_path_maybe = []  # want to keep track of this so we know whether or not to delete it
    # try:
    # Make a temporary folder to hold the temporary DLC folder
    # (want to have them for different instances running on same
    # node without collisions)
    with tempfile.TemporaryDirectory(prefix=generalized_slash_tmp_path + "/") as scratch_folder_path:
        #scratch_dlc_container_path = tempfile.mkdtemp(prefix=generalized_tmp_folder_path + "/")
        #scratch_dlc_container_path_maybe = [scratch_dlc_container_path]
        print("scratch_folder_path is %s" % scratch_folder_path)

        # Determine the absolute path to the temporary DLC folder
        scratch_dlc_root_folder_path = os.path.join(scratch_folder_path, "dlc-working")
        print("scratch_dlc_root_folder_path is %s" % scratch_dlc_root_folder_path)

        # Copy the reference DLC folder to the scratch one
        shutil.copytree(template_dlc_root_folder_path, scratch_dlc_root_folder_path)

        # Load the configuration file
        configuration_file_name = 'myconfig.py'
        configuration_file_path = os.path.join(targets_folder_path, configuration_file_name)
        configuration = dlct.load_configuration_file(configuration_file_path)

        # Copy the configuration file into the scratch DLC folder
        scratch_configuration_file_path = os.path.join(scratch_dlc_root_folder_path, configuration_file_name)
        shutil.copyfile(configuration_file_path, scratch_configuration_file_path)

        # Determine whether the targets folder contains subfolders.
        # If not, we assume it's old-style, and contains .csv and .png files.
        # If the targets folder contains subfolders, we assume it new-style, and that
        # the subfolders each contain .csv and .png files.
        target_subfolder_names = [ dir_entry.name for dir_entry in os.scandir(targets_folder_path) if dir_entry.is_dir() ]
        is_new_style = bool(target_subfolder_names)  # true iff nonempty

        # Copy the targets folder to the scratch DLC folder, making a containing folder if needed
        task = configuration['Task']
        data_folder_name = 'data-' + task  # e.g. "data-licking-side"
        if is_new_style :
            print('Targets folder is new-style, with multiple subfolders')
            scratch_targets_folder_path = os.path.join(scratch_dlc_root_folder_path,
                                                       'Generating_a_Training_Set',
                                                       data_folder_name)
        else:
            # Is old-style, have to create a subfolder to hold the .csv's and .png's
            print("Targets folder is old-style, with images and .csv's right in it")
            targets_folder_name = os.path.basename(targets_folder_path)
            scratch_targets_folder_path = os.path.join(scratch_dlc_root_folder_path,
                                                       'Generating_a_Training_Set',
                                                       data_folder_name,
                                                       targets_folder_name)
        shutil.copytree(targets_folder_path, scratch_targets_folder_path)

        # To be on the safe side, remove the myconfig.py file from the scratch targets folder, since it' not needed there
        configuration_file_within_scratch_targets_folder_path = os.path.join(scratch_targets_folder_path, configuration_file_name)
        os.remove(configuration_file_within_scratch_targets_folder_path)

        # Determine absolute path the the scratch Generating_a_Training_Set folder
        training_folder_path = os.path.join(scratch_dlc_root_folder_path, "Generating_a_Training_Set")
        print("training_folder_path: %s\n" % training_folder_path)

        # cd into the scratch Generating_a_Training_Set folder
        with dlct.Chdir(training_folder_path):
            # Determine absolute path to the Step2 script
            make_data_frame_script_path = os.path.join(delectable_folder_path,
                                                       "dlc",
                                                       "Generating_a_Training_Set",
                                                       "Step2_ConvertingLabels2DataFrame.py")

            # Run the Step 2 script
            return_code = os.system('%s %s' % (python_executable_path, make_data_frame_script_path))
            if return_code != 0:
                raise RuntimeError(
                    'There was a problem running, %s return code %d' % (make_data_frame_script_path, return_code))

            # Determine absolute path to the Step3 script
            check_labels_script_path = os.path.join(delectable_folder_path,
                                                    "dlc",
                                                    "Generating_a_Training_Set",
                                                    "Step3_CheckLabels.py")

            # Run the Step3 script
            return_code = os.system('%s %s' % (python_executable_path, check_labels_script_path))
            if return_code != 0:
                raise RuntimeError(
                    'There was a problem running, %s return code %d' % (check_labels_script_path, return_code))

            # Determine absolute path to the Step 4 script
            make_training_file_script_path = os.path.join(delectable_folder_path,
                                                          "dlc",
                                                          "Generating_a_Training_Set",
                                                          "Step4_GenerateTrainingFileFromLabelledData.py")
            print("make_training_file_script_path: %s\n" % make_training_file_script_path)

            # Run the Step 4 script
            return_code = os.system('%s %s' % (python_executable_path, make_training_file_script_path))
            if return_code != 0:
                raise RuntimeError(
                    'There was a problem running %s, return code %d' % (make_training_file_script_path, return_code))

        # Copy the relevant folders over to the pose-tensorflow folder
        scratch_tensorflow_models_path = os.path.join(scratch_dlc_root_folder_path,
                                                      "pose-tensorflow",
                                                      "models")
        date = configuration['date']
        trainset_folder_name = task + date + "-trainset95shuffle1"
        source_trainset_folder_path = os.path.join(scratch_dlc_root_folder_path, "Generating_a_Training_Set",
                                                   trainset_folder_name)
        dest_trainset_folder_path = os.path.join(scratch_tensorflow_models_path, trainset_folder_name)
        unaugmented_data_set_folder_name = "UnaugmentedDataSet_" + task + date
        unaugmented_data_set_folder_path = os.path.join(scratch_dlc_root_folder_path, "Generating_a_Training_Set",
                                                        unaugmented_data_set_folder_name)
        dest_unaugmented_data_set_folder_path = os.path.join(scratch_tensorflow_models_path, unaugmented_data_set_folder_name)
        shutil.copytree(source_trainset_folder_path, dest_trainset_folder_path)
        shutil.copytree(unaugmented_data_set_folder_path, dest_unaugmented_data_set_folder_path)

        # Copy the pretrained models to the right place in the scratch folder
        scratch_pretrained_folder_path = os.path.join(scratch_tensorflow_models_path,
                                                      'pretrained')
        if not os.path.exists(scratch_pretrained_folder_path):
            os.makedirs(scratch_pretrained_folder_path)
        resnet_50_path = os.path.join(delectable_folder_path, 'resnet_v1_50.ckpt')
        resnet_101_path = os.path.join(delectable_folder_path, 'resnet_v1_101.ckpt')
        shutil.copy(resnet_50_path, scratch_pretrained_folder_path)
        shutil.copy(resnet_101_path, scratch_pretrained_folder_path)

        # cd into the scratch training folder
        folder_for_running_training_path = os.path.join(scratch_tensorflow_models_path, trainset_folder_name, "train")
        with dlct.Chdir(folder_for_running_training_path):
            # Run the training script (takes a long time)
            training_script_path = os.path.join(delectable_folder_path, "dlc", "pose-tensorflow", "train.py")
            os.system('%s %s' % (python_executable_path, training_script_path))

            # Delete the pretrained model folder in the scratch area
            shutil.rmtree(scratch_pretrained_folder_path)    

        # Copy the scratch model folder to the final output folder location
        print("About to copy result to final location...")
        print("model_folder_path: %s" % model_folder_path)
        shutil.copytree(scratch_tensorflow_models_path, model_folder_path)

        # Copy the config file to the output folder location also.
        # This is useful to have around for later operations taking the model as input.
        shutil.copy(configuration_file_path, os.path.join(model_folder_path, 'myconfig.py'))


#
# main
#
if __name__ == "__main__":
    # Get the arguments    
    targets_folder_path = os.path.abspath(sys.argv[1])
    model_folder_path = os.path.abspath(sys.argv[2])

    # Run the training
    train_model(targets_folder_path, model_folder_path)
