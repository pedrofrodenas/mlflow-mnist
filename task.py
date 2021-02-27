#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:31:06 2020

@author: pedrofrodenas
"""

from config import Config
from generators import TFRecordsGenerator

from model_distribute import TrainMNIST

class MNISTConfig(Config):
    """Configuration for training MNIST dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "mnist"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 64
    
    

tf_data_train = TFRecordsGenerator(MNISTConfig.TRAIN_DIR)
tf_data_val = TFRecordsGenerator(MNISTConfig.TEST_DIR)
    
mnist_trainer = TrainMNIST(MNISTConfig)

mnist_trainer.train(tf_data_train, tf_data_val, 
                    learning_rate = 0.002, 
                    epochs = 5)


















