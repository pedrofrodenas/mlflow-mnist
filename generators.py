#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 17:50:16 2021

@author: pedrofRodenas
"""

import os
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

class TFRecordsGenerator():
    
    def __init__(self, tfrecords_dir):
        
        # List all *.tfrecord files for the selected split
        pattern = os.path.join(tfrecords_dir, '*.tfrec')
        files_ds = tf.data.Dataset.list_files(pattern)
        
        # Get number of samples in dataset in order to shuffle it later
        # firstly get the first file path to tfrecords. The number of
        # samples is codified in tfrecords file name after '--' separator.
        iterator = files_ds.as_numpy_iterator()
        element = next(iterator)
        element = element.decode('utf-8')
        # Getrid of tfrecords path, keep name
        element = os.path.basename(element)
        
        # Parse number of samples in dataset ()
        self.n_samples = int(os.path.splitext(element.split('--')[-1])[0])

        # Disregard data order in favor of reading speed
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
    
        # Read TFRecord files in an interleaved order
        self.ds = tf.data.TFRecordDataset(files_ds,
                                          compression_type='ZLIB',
                                          num_parallel_reads=AUTOTUNE)
        
    def __len__(self):
        return self.n_samples
        
    def _parse_batch(self, record_batch):
    
        # Create a description of the features
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
    
        # Parse the input `tf.Example` proto using the dictionary above
        example = tf.io.parse_example(record_batch, feature_description)
        
        example['image'] = tf.io.decode_image(example['image'], 
                                              dtype=tf.dtypes.float32)
        
        example['image'] = example['image']
    
        return example['image'], example['label']
    
    def prepare(self, batch_size):
        
        # Reshule dataset after each full samples iteration
        self.ds = self.ds.shuffle(buffer_size=self.n_samples,
                                  reshuffle_each_iteration=True)
        
        self.ds = self.ds.map(self._parse_batch, num_parallel_calls=AUTOTUNE)
        
        # Prepare batches, drop remainded samples when pass through the whole
        # dataset. This is useful in order to calculate later steps_per_epochs
        # accurately
        self.ds = self.ds.batch(batch_size, drop_remainder=True)
        
        return self.ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        
        
        
        
        
        
        