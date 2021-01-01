#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:05:52 2020

@author: pedrofrodenas
"""

from dataset import TFRecordsConverter
from mnist_to_disk import SaverMNIST
from config import Config


mnist_to_disk = SaverMNIST(Config.TRAIN_IMAGES_DIR, Config.TEST_IMAGES_DIR, 
                           Config.TRAIN_CSV_PATH, Config.TEST_CSV_PATH)

mnist_to_disk.run()

# train_dataset_converter = TFRecordsConverter(csv_path=Config.TRAIN_CSV_PATH,
#                                              split_name='train',
#                                              output_dir=Config.DATASET_DIR,
#                                              image_shape=Config.IMAGE_SHAPE,
#                                              n_shards=2)

# shard_splits = train_dataset_converter.convert()

