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


