#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:05:38 2020

@author: pedrofRodenas
"""

import os
 
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

      
def mnist_writer(save_path):
    
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Encode array to .png
    x_train_encoded = np.array(list(map(image_to_encodedbytes, x_train)))
    x_test_encoded = np.array(list(map(image_to_encodedbytes, x_test)))

    # Convert array to pandas dataframe
    df_train = pd.DataFrame({'images':x_train_encoded, 'labels':y_train})
    df_test = pd.DataFrame({'images':x_test_encoded, 'labels':y_test})
    
    # Build save path
    train_save_name = os.path.join(save_path, 'train.csv')
    train_save_name = os.path.normpath(train_save_name)
    test_save_name = os.path.join(save_path, 'test.csv')
    test_save_name = os.path.normpath(test_save_name)
     
    # Save dataframe
    df_train.to_csv(train_save_name)
    df_test.to_csv(test_save_name)
        
        
def image_to_encodedbytes(image):
    success, encoded_image = cv2.imencode('.png', image)
    data = encoded_image.tobytes()
    return data