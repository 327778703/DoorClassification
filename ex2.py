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
from CustomMetrics_ex import ConfusionMatrixMetric

a = tf.constant([[0.1, 0.1, 0.7, 0.1], [0.6, 0.1, 0.1, 0.2], [0.4, 0.1, 0.2, 0.3],
                 [0.1, 0.1, 0.7, 0.1], [0.6, 0.1, 0.1, 0.2], [0.4, 0.1, 0.2, 0.3],
                 [0.1, 0.1, 0.7, 0.1], [0.2, 0.1, 0.1, 0.6], [0.4, 0.1, 0.2, 0.3]])
b = tf.constant([2, 1, 0, 2, 3, 1, 0, 3, 0])

cm = ConfusionMatrixMetric(4)
acc = keras.metrics.SparseCategoricalAccuracy()
cm.update_state(b, a)
acc.update_state(b, a)
print("1:", cm.total_cm)
cm.result()
print(acc.result())
c = tf.constant([[0.2, 0, 0.7, 0.1], [0.6, 0.1, 0.1, 0.2], [0.3, 0.1, 0.4, 0.3],
                 [0.1, 0.7, 0.1, 0.1], [0.2, 0.1, 0.1, 0.6], [0.4, 0.1, 0.2, 0.3],
                 [0.1, 0.1, 0.7, 0.1], [0.2, 0.1, 0.1, 0.6], [0.3, 0.1, 0.2, 0.4]])
d = tf.constant([2, 1, 2, 2, 3, 1, 0, 3, 0])

cm.update_state(d, c)
acc.update_state(d, c)
print(acc.result())
print("2:", cm.total_cm)
cm.result()



