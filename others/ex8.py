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
from MyModel import MyModel
from CustomMetrics import TopkAccuracy, ConfusionMatrixMetric

a = tf.constant([[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.7, 0.1, 0.2]])
b = tf.constant([0, 2, 2])
top = TopkAccuracy(2)
acc = keras.metrics.SparseCategoricalAccuracy()
top.update_state(b, a)
acc.update_state(b, a)
print(acc.result())
c = tf.nn.in_top_k(b, a, 2)
print(c)
print(tf.math.count_nonzero(c) / tf.cast(tf.size(c), tf.int64))
print(top.topk)
