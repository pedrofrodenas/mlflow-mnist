#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:05:05 2021

@author: pedrofRodenas
"""

import tensorflow as tf


class NBatchLogger(tf.keras.callbacks.Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            metrics_log = ''
            for k in self.params['metrics']:
                if k in logs:
                    val = logs[k]
                    if abs(val) > 1e-3:
                        metrics_log += ' - %s: %.4f' % (k, val)
                    else:
                        metrics_log += ' - %s: %.4e' % (k, val)
            print('{}/{} ... {}'.format(self.seen,
                                        self.params['samples'],
                                        metrics_log))
            
            
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        print('Hola Mundo')
        
    def on_train_begin(self, logs=None):
        print('Empieza el entreno')
        
    def on_train_batch_begin(self, batch, logs=None):
        print('Empieza el batch {0}'.format(batch))
        
    def on_epoch_begin(self, epoch, logs=None):
        print("Empieza la epoca {0}".format(epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        print("Termina la epoca {0}".format(epoch))
        
    # def on_predict_batch_begin(self, batch, logs=None):
        