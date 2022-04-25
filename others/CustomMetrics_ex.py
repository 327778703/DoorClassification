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


class TopkAccuracy(keras.metrics.Metric):
    def __init__(self, k=2, name='top-2_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.topk = self.add_weight(name="topk", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1])
        correct = tf.nn.in_top_k(y_true, y_pred, self.k)
        top_k_accuracy = tf.math.count_nonzero(correct) / tf.cast(tf.size(correct), tf.int64)
        top_k_accuracy = tf.cast(top_k_accuracy, tf.float32)

        # values, indices = tf.math.top_k(y_pred, k=self.k, sorted=True)
        # # print(values, indices)
        # y = tf.reshape(y_true, [-1, 1])
        # # print(y)
        # correct = tf.cast(tf.equal(y, indices), tf.float32)
        # # print(correct)
        # top_k_accuracy = tf.reduce_mean(correct) * self.k
        self.topk.assign(top_k_accuracy)

    def result(self):
        return self.topk

    def reset_states(self):
        self.topk.assign(0.0)

class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    """
    A custom Keras metric to compute the running average of the confusion matrix
    """

    def __init__(self, num_classes, **kwargs):
        super(ConfusionMatrixMetric, self).__init__(name='confusion_matrix_metric',
                                                    **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes, num_classes), initializer="zeros")

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        return self.total_cm

    def result(self):
        return self.process_confusion_matrix()

    def confusion_matrix(self, y_true, y_pred):
        y_pred = tf.argmax(y_pred, 1)
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        return cm

    def process_confusion_matrix(self):
        cm = self.total_cm
        diag_part = tf.linalg.diag_part(cm)
        precision = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        recall = diag_part / (tf.reduce_sum(cm, 1) + tf.constant(1e-15))
        f1 = 2 * precision * recall / (precision + recall + tf.constant(1e-15))
        print('''diag_part:{}
        precision:{}
        recall:{}
        f1:{}
        '''.format(diag_part, precision, recall, f1))
        # precision = tf.reduce_mean(precision)
        # recall = tf.reduce_mean(recall)
        # f1 = tf.reduce_mean(f1)

        return precision, recall, f1

    def fill_output(self, output):
        results = self.result()
        output['precision'] = results[0]
        output['recall'] = results[1]
        output['F1'] = results[2]
