#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 13:05:52 2020

@author: pedrofrodenas
"""

from dataset import TFRecordsConverter
from generators import TFRecordsGenerator
from mnist_to_disk import SaverMNIST
from config import Config


mnist_to_disk = SaverMNIST(Config.TRAIN_IMAGES_DIR, Config.TEST_IMAGES_DIR, 
                           Config.TRAIN_CSV_PATH, Config.TEST_CSV_PATH)

#mnist_to_disk.run()

train_dataset_converter = TFRecordsConverter(csv_path=Config.TRAIN_CSV_PATH,
                                             image_dir=Config.TRAIN_IMAGES_DIR,
                                             split_name='train',
                                             output_dir=Config.TRAIN_DIR,
                                             image_shape=Config.IMAGE_SHAPE,
                                             n_shards=2)

test_dataset_converter = TFRecordsConverter(csv_path=Config.TEST_CSV_PATH,
                                            image_dir=Config.TEST_IMAGES_DIR,
                                            split_name='test',
                                            output_dir=Config.TEST_DIR,
                                            image_shape=Config.IMAGE_SHAPE)

train_dataset_converter.convert()
test_dataset_converter.convert()


train_dataset = TFRecordsGenerator(Config.TRAIN_DIR)

ds = train_dataset.prepare(batch_size=8)



