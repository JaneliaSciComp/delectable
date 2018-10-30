#! /usr/bin/python3

import sys
import os
import tempfile
import shutil
import pwd


def get_username():
    return pwd.getpwuid(os.getuid())[0]


def determine_scratch_folder_path():
    # If a /scratch folder exists, use that.  Otherwise, use /tmp.
    username = get_username()
    my_scratch_folder_path = '/scratch/%s' % username
    if os.path.isdir(my_scratch_folder_path) :
        return my_scratch_folder_path
    else:
        return '/tmp'

def is_empty(list) :
    return len(list)==0

def file_name_without_extension_from_path(path):
    file_name = os.path.basename(path)
    return os.path.splitext(file_name)[0]

def does_match_extension(file_name, target_extension) :
    # target_extension should include the dot
    extension = os.path.splitext(file_name)[1]
    return (extension == target_extension)

def replace_extension(file_name, new_extension) :
    # new_extension should include the dot
    base_name = os.path.splitext(file_name)[0]
    return base_name + new_extension

class add_path:
    def __init__(self, path):
        self.original_sys_path = sys.path.copy()
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        sys.path = self.original_sys_path.copy()

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
