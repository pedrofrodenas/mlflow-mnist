#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 13:04:07 2021

@author: pedrofrodenas
"""

import errno
import os


def make_dirs(path_list):
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)

def make_containing_dirs(path_list):
    
    for path in path_list:
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)