#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:05:38 2020

@author: pedrofRodenas
"""

import file_utils

import os
 
import pandas as pd
from PIL import Image
import tensorflow as tf

    
class SaverMNIST():
    def __init__(self, image_train_path, image_test_path, csv_train_path, 
                 csv_test_path):
        
        self._image_format = '.png'
        
        self.store_image_paths = [image_train_path, image_test_path]
        self.store_csv_paths = [csv_train_path, csv_test_path]
        
        file_utils.make_dirs(self.store_image_paths)
        file_utils.make_containing_dirs(self.store_csv_paths)
             
        # Load MNIST dataset
        mnist = tf.keras.datasets.mnist
        self.data = mnist.load_data()
        
    def run(self):
        
        for collection, store_image_path, store_csv_path in zip(self.data, 
                                                                self.store_image_paths,
                                                                self.store_csv_paths):
            
            labels_list = []
            paths_list = []
            
            for index, (image, label) in enumerate(zip(collection[0], 
                                                       collection[1])):
                im = Image.fromarray(image)
                width, height = im.size
                image_name = str(index) + self._image_format
                
                # Build save path
                save_path = os.path.join(store_image_path, image_name)
                im.save(save_path)
                
                labels_list.append(label)
                paths_list.append(save_path)
                
            df = pd.DataFrame({'image_paths':paths_list, 'labels': labels_list})
            
            df.to_csv(store_csv_path)
                