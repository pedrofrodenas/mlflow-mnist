#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 21:25:42 2020

@author: pedrofRodenas
"""

import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf

_SEED = 2020

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

class TFRecordsConverter:
    """Convert WAV files to TFRecords."""

    # When compression is used, resulting TFRecord files are four to five times
    # smaller. So, we can reduce the number of shards by this factor
    _COMPRESSION_SCALING_FACTOR = 4

    def __init__(self, csv_path , image_dir, split_name ,output_dir, 
                 image_shape ,n_shards=None):
        
        self.image_dir = image_dir
        self.split_name = split_name
        self.output_dir = output_dir
        self.image_shape = image_shape
        
        df = pd.read_csv(csv_path, index_col=0)
        # Shuffle data by "sampling" the entire data-frame
        self.df = df.sample(frac=1, random_state=_SEED)
        
        self.n_samples = len(self.df)

        if n_shards is None:
            self.n_shards = self._n_shards()
        else:
            self.n_shards =  n_shards

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def __repr__(self):
        return ('{}.{}(output_dir={}, n_shards={}, n_samples={}').format(
            self.__class__.__module__,
            self.__class__.__name__,
            self.output_dir,
            self.n_shards,
            self.n_samples
        )

    def _n_shards(self):
        """Compute number of shards for number of samples.
        TFRecords are split into multiple shards. Each shard's size should be
        between 100 MB and 200 MB according to the TensorFlow documentation.
        Parameters
        ----------
        n_samples : int
            The number of samples to split into TFRecord shards.
        Returns
        -------
        n_shards : int
            The number of shards needed to fit the provided number of samples.
        """
        return math.ceil(self.n_samples / self._shard_size())

    def _shard_size(self):
        """Compute the shard size.
        Computes how many WAV files with the given sample-rate and duration
        fit into one TFRecord shard to stay within the 100 MB - 200 MB limit.
        Returns
        -------
        shard_size : int
            The number samples one shard can contain.
        """
        shard_max_bytes = 200 * 1024**2  # 200 MB maximum
        bytes_per_sample = self.image_shape[0] * self.image_shape[1] + 1 # label byte (uint8)
        shard_size = shard_max_bytes // bytes_per_sample
        return shard_size * self._COMPRESSION_SCALING_FACTOR
            

    def _write_tfrecord_file(self, shard_data):
        """Write TFRecord file.
        Parameters
        ----------
        shard_data : tuple (str, list)
            A tuple containing the shard path and the list of indices to write
            to it.
        """
        for shard_path, indices in shard_data:
            print(shard_path)
            with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
                for index in indices:
                    file_path = self.df.image_paths.iloc[index]
                    label = self.df.labels.iloc[index]
                    
                    with open(file_path, "rb") as image:
                        image_bytearray = image.read() 
    
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'image': _bytes_feature(image_bytearray),
                        'label': _int64_feature(label)}))
    
                    out.write(example.SerializeToString())

    def _get_shard_path(self, split, shard_id, shard_size):
        """Construct a shard file path.
        Parameters
        ----------
        split : str
            The data split. Typically 'train', 'test' or 'validate'.
        shard_id : int
            The shard ID.
        shard_size : int
            The number of samples this shard contains.
        Returns
        -------
        shard_path : str
            The constructed shard path.
        """
        return os.path.join(self.output_dir,
                            '{}-{:03d}-{}--{}.tfrec'.format(split, shard_id,
                                                            shard_size,
                                                            self.n_samples))

    def _split_data_into_shards(self):
        """Split data into train/test/val sets.
        Split data into training, testing and validation sets. Then,
        divide each data set into the specified number of TFRecords shards.
        Returns
        -------
        shards : list [tuple]
            The shards as a list of tuples. Each item in this list is a tuple
            which contains the shard path and a list of indices to write to it.
        """
        shards = []
        
        split = self.split_name
        size = self.n_samples
        n_shards = self.n_shards

        offset = 0
        
        print('Splitting {} set into TFRecord shards...'.format(split))
        shard_size = math.ceil(size / n_shards)
        cumulative_size = offset + size
        for shard_id in range(1, n_shards + 1):
            step_size = min(shard_size, cumulative_size - offset)
            shard_path = self._get_shard_path(split, shard_id, step_size)
            # Select a subset of indices to get only a subset of
            # audio-files/labels for the current shard.
            file_indices = np.arange(offset, offset + step_size)
            shards.append((shard_path, file_indices))
            offset += step_size

        return shards

    def convert(self):
        """Convert to TFRecords."""
        shard_splits = self._split_data_into_shards()

        self._write_tfrecord_file(shard_splits)

        print('Number of examples: {}'.format(self.n_samples))
        print('TFRecord files saved to {}'.format(self.output_dir))
        
        
        
        
        
        
        
        
        
        
        
        
        