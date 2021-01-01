#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:45:51 2020

@author: pedrofrodenas
"""

import os


class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None

    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1
    
    DATASET_DIR = "dataset"
    
    TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
    TEST_DIR = os.path.join(DATASET_DIR, 'test')
    
    TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
    TEST_IMAGES_DIR = os.path.join(TEST_DIR, 'images')
    
    TRAIN_CSV_PATH = os.path.join(TRAIN_DIR, 'train.csv')
    TEST_CSV_PATH = os.path.join(TEST_DIR, 'test.csv')
    
    IMAGE_SHAPE = (28, 28)


    def __init__(self):
        """Set values of computed attributes."""


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")