#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:33:20 2020

@author: pedrofRodenas
"""

import tensorflow as tf
import numpy as np


class Trainer():
    
    def __init__(self, model, optimizer, loss_object, test_loss,
                 train_accuracy, test_accuracy,epochs, images_per_gpu, 
                 steps_per_epoch_train, steps_per_epoch_val, 
                 strategy, writer ,callbacks=None):
        
        self.model = model
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.test_loss = test_loss
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.epochs = epochs
        self.images_per_gpu = images_per_gpu
        self.steps_per_epoch_train = steps_per_epoch_train
        self.steps_per_epoch_val = steps_per_epoch_val
        self.strategy = strategy
        self.writer = writer
        self.callbacks = callbacks
        
        self.logs = {}
        
    def _write_logs(self, logs, index):

        with self.writer.as_default():                                                                                                                                                                                                                               
            for name, value in logs.items():                                                                                                                                                                                                                              
                if name in ['batch', 'size']:                                                                                                                                                                                                   
                    continue                                                                                                                                                                          
                if isinstance(value, np.ndarray):
                    tf.summary.scalar(name, value.item(), step=index)                                                                                                                                                                                               
                else:
                    tf.summary.scalar(name, value, step=index)
            self.writer.flush()
        
    def compute_loss(self, label, predictions):
        
        # loss = (loss_object + model_losses) / images_per_gpu * num_replicas_in_sync
        # loss = (loss_object + model_losses) / global_batch_size
        loss = tf.reduce_sum(self.loss_object(label, predictions)) * (
            1. / self.images_per_gpu)
        loss += (sum(self.model.losses) * 1. / self.strategy.num_replicas_in_sync)
        return loss

    def train_step(self, inputs):
      """One train step.
      Args:
        inputs: one batch input.
      Returns:
        loss: Scaled loss.
      """
  
      image, label = inputs
      with tf.GradientTape() as tape:
        predictions = self.model(image, training=True)
        loss = self.compute_loss(label, predictions)
      gradients = tape.gradient(loss, self.model.trainable_variables)
      self.optimizer.apply_gradients(zip(gradients,
                                         self.model.trainable_variables))
  
      self.train_accuracy(label, predictions)
      return loss
  
    def test_step(self, inputs):
      """One test step.
      Args:
        inputs: one batch input.
      """
      image, label = inputs
      predictions = self.model(image, training=False)
  
      unscaled_test_loss = self.loss_object(label, predictions) + sum(
          self.model.losses)
  
      self.test_accuracy(label, predictions)
      self.test_loss(unscaled_test_loss)
      
      
    # `run` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(self, dataset_inputs):
        per_replica_losses = self.strategy.run(self.train_step, args=(dataset_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)
    
    @tf.function
    def distributed_test_step(self, dataset_inputs):
        return self.strategy.run(self.test_step, args=(dataset_inputs,))
    

    def custom_loop(self, train_dist_dataset, test_dist_dataset):
        """Custom training and testing loop.
        Args:
          train_dist_dataset: Training dataset created using strategy.
          test_dist_dataset: Testing dataset created using strategy.
          strategy: Distribution strategy.
        """
        
        for cb in self.callbacks_list:
            cb.on_train_begin()
        
        # TRAIN LOOP
        for epoch in range(self.epochs):
            
            for cb in self.callbacks_list:
                cb.on_epoch_begin(epoch)  
            
            total_loss = 0.0
            num_batches = 0
            for x in train_dist_dataset:
                
                for cb in self.callbacks_list:
                    cb.on_train_batch_begin(num_batches, self.logs)
                    
                total_loss += self.distributed_train_step(x)
                num_batches += 1
                
                for cb in self.callbacks_list:
                    cb.on_train_batch_end(num_batches, self.logs)

            train_loss = total_loss / tf.cast(num_batches, dtype=tf.float32)
          
            # TEST LOOP
            for x in test_dist_dataset:
                self.distributed_test_step(x)
          
          
            template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                        "Test Accuracy: {}")
            print (template.format(epoch+1, train_loss,
                                   self.train_accuracy.result(), 
                                   self.test_loss.result(),
                                   self.test_accuracy.result()))
            
            self.logs.update({'loss': train_loss,
                              'test_loss': self.test_loss.result(),
                              'train_accuracy': self.train_accuracy.result().numpy(),
                              'test_accuracy': self.test_accuracy.result().numpy()})
            
            self._write_logs(self.logs, epoch)
          
            self.test_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_accuracy.reset_states()
            
            for cb in self.callbacks_list:
                cb.on_epoch_end(epoch, self.logs)
                
        for cb in self.callbacks_list:
            cb.on_train_end(self.logs)
        self.writer.close()

