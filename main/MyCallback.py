# -*- coding: utf-8 -*-
# basic_model，未标准化，使用数据增强，使用VGG16，全局平均，2个FC+Dropout(0.5)

import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
from keras_preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.rc("font", family='FangSong')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from PIL import Image
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory

class MyCallback():
    def CheckPointCallback(self, path=r'D:\MyFiles\ResearchSubject\door4\doorModels/VGG16-ckpt-512/avg'):
        os.makedirs(path, exist_ok=True)
        CheckPoint_Path = path + '/cp-{epoch:03d}-{loss:.3f}-{acc:.3f}-{top-2_acc:.3f}-{F1:.3f}-{val_loss:.3f}-' \
                             '{val_acc:.3f}-{val_top-2_acc:.3f}-{val_F1:.3f}.ckpt'
        cp_callback = keras.callbacks.ModelCheckpoint(CheckPoint_Path, verbose=2, save_weights_only=True)
        return cp_callback

    def TensorboardCallback(self, log_dir=r'D:\MyFiles\ResearchSubject\door4\doorTensorboard/VGG16-ckpt-512/avg'):
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
        return tensorboard_callback

    def decay(self, epoch):
        step_size = 1600
        iterations = epoch * 41
        base_lr = 1e-05
        max_lr = 3.65e-04
        cycle = np.floor(1 + iterations / (2 * step_size))
        x = np.abs(iterations / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * x
        return lr

    def LearningRateCallback(self):
        learningrate_callback = tf.keras.callbacks.LearningRateScheduler(self.decay)
        return learningrate_callback
