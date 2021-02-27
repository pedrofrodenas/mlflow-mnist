#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:49:31 2020

@author: pedrofrodenas
"""

from callbacks import CustomCallback
from trainer import Trainer

import datetime
import os

import tensorflow as tf

class TrainMNIST():
    def __init__(self, config):
        
        tf.keras.backend.clear_session()
        
        self.config = config
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
        
        self.global_batch_size = self.config.IMAGES_PER_GPU * self.strategy.num_replicas_in_sync
        
        self.steps_per_epochs_train = len(tf_data_train) // self.global_batch_size
        self.steps_per_epochs_val = len(tf_data_val) // self.global_batch_size
        
        ds_train = tf_data_train.prepare(self.global_batch_size)
        ds_val = tf_data_val.prepare(self.global_batch_size)
        
        with self.strategy.scope():
            
            train_dist_dataset = self.strategy.experimental_distribute_dataset(ds_train)
            val_dist_dataset = self.strategy.experimental_distribute_dataset(ds_val)
        
            self.keras_model = self.build()
            
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE)
            
            test_loss = tf.keras.metrics.Mean(name='test_loss')

            train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name='train_accuracy')
            test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
                name='test_accuracy')
            
            # Queda definir el learning_rate
            optimizer = tf.keras.optimizers.Adam()
            
            # Callback list definition
            callback_list = []
            
            callback_list.append(CustomCallback())
            
            self.trainer = Trainer(self.keras_model,
                                   optimizer,
                                   loss_object,
                                   test_loss,
                                   train_accuracy,
                                   test_accuracy,
                                   self.config.EPOCHS, 
                                   self.config.IMAGES_PER_GPU, 
                                   self.steps_per_epochs_train,
                                   self.steps_per_epochs_val,
                                   self.strategy)
            
            return train_dist_dataset, val_dist_dataset
        
    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """

        now = datetime.datetime.now()


        # Directory for training logs
        self.model_dir = os.path.join(self.config.LOG_DIR, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.model_dir, "mnist_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        
        # Define save model name
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")
        
    def train(self, tf_data_train, tf_data_val, learning_rate, epochs):
        """Train the model.
        
        """

        # Create log_dir if it does not exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

       
        train_dist_dataset, val_dist_dataset = self.compile(tf_data_train, 
                                                            tf_data_val)
        
        self.trainer.custom_loop(train_dist_dataset, val_dist_dataset)

       
        
            
            
        
        
        
        
        
    
    