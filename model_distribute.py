#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:49:31 2020

@author: pedrofrodenas
"""

import tensorflow as tf

class TrainMNIST():
    def __init__(self, config, log_dir):
        
        tf.keras.backend.clear_session()
        
        self.config = config
        self.log_dir = log_dir
        
        self.keras_model = None
    
    def build(self):
        
        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        return model
    
    def compile(self, tf_data_train , tf_data_val):
        
        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
        
        self.global_batch_size = self.config.IMAGES_PER_GPU * self.strategy.num_replicas_in_sync()
        
        self.steps_per_epochs_train = len(tf_data_train) // self.global_batch_size
        self.steps_per_epochs_val = len(tf_data_val) // self.global_batch_size
        
        ds_train = tf_data_train.create(self.global_batch_size)
        ds_val = tf_data_val.create(self.global_batch_size)
        
        with self.strategy.scope():
            
            train_dist_dataset = self.strategy.experimental_distribute_dataset(ds_train)
            val_dist_dataset = self.strategy.experimental_distribute_dataset(ds_val)
        
            self.keras_model = self.build()
            
            # Callback list definition
            callback_list = []
        
            # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, 
            #                                    beta_1=0.9, 
            #                                    beta_2=0.999, 
            #                                    epsilon=1e-07, 
            #                                    amsgrad=False,
            #                                    name='Adam',
            #                                    )
            
            return train_dist_dataset, val_dist_dataset
            
            
        
        
        
        
        
    
    