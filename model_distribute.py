#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:49:31 2020

@author: pedrofrodenas
"""

import datetime
import os

import tensorflow as tf

class TrainMNIST():
    def __init__(self, config, log_dir):
        
        tf.keras.backend.clear_session()
        
        self.config = config
        self.log_dir = log_dir
        self.set_log_dir()
        
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
        
    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()


        # Directory for training logs
        self.model_dir = os.path.join(self.log_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.model_dir, "mnist_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
        
    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        
        """

        # Create log_dir if it does not exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

       
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

       

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)
            
            
        
        
        
        
        
    
    