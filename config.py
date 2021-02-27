#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:45:51 2020

@author: pedrofrodenas
"""

import os
import yaml

class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None
    
    IMAGES_PER_GPU = 32
  
    LOG_DIR = "logs"
    
    DATASET_DIR = "dataset"
    
    TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
    TEST_DIR = os.path.join(DATASET_DIR, 'test')
    
    TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
    TEST_IMAGES_DIR = os.path.join(TEST_DIR, 'images')
    
    TRAIN_CSV_PATH = os.path.join(TRAIN_DIR, 'train.csv')
    TEST_CSV_PATH = os.path.join(TEST_DIR, 'test.csv')
    
    IMAGE_SHAPE = (28, 28)
    
    EPOCHS = 10


    def __init__(self):
        """Set values of computed attributes."""


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
    def import_config(self, config_path):
        """Reads configuration variables
        from .yaml file and set the variables
        """
        
        # Necesary for importing python tuples from
        # .yaml
        PrettySafeLoader.add_constructor(
            u'tag:yaml.org,2002:python/tuple',
            PrettySafeLoader.construct_python_tuple)
        with open(config_path, 'r') as stream:
            try:
                config_dict = yaml.load(stream, Loader=PrettySafeLoader)
            except yaml.YAMLError as exc:
                print(exc)
         
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
        
           
    def export(self, folder):
        """Writes Configuration values
        onto a .yaml file"""
        config = {}
           
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                config[a] = getattr(self, a)
        with open(os.path.join(folder, 'configuration.yaml'), 'w+') as outfile:
            yaml.dump(config, outfile, allow_unicode=True)