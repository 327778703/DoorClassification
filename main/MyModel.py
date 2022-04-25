# -*- coding: utf-8 -*-
# 两个512全连接层，无标准化，对应models:VGG16-ckpt-512

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

class MyModel():
    def __init__(self, inputs):
        self.inputs = inputs
        self.CreateMyModel()

    def CreateMyModel(self):
        data_augmentation = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomFlip()
        ])
        preprocess_input = keras.applications.vgg16.preprocess_input
        base_model = keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet',
                                              pooling='avg')
        # a = base_model.layers[-1].output
        # base_model.summary()
        base_model.trainable = False
        fc1 = keras.layers.Dense(512, activation='relu', name='dense_1')
        drop1 = keras.layers.Dropout(0.5, name='dropout_1')
        fc2 = keras.layers.Dense(512, activation='relu', name='dense_2')
        drop2 = keras.layers.Dropout(0.5, name='dropout_2')
        fc3 = keras.layers.Dense(64, activation='softmax', name='output')

        x = preprocess_input(self.inputs)
        x = data_augmentation(x)
        x = base_model(x)
        x = fc1(x)
        x = drop1(x)
        x = fc2(x)
        x = drop2(x)
        self.outputs = fc3(x)
        return CustomModel(self.inputs, self.outputs)

class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, y_pred)
        # print(self.metrics_names)

        output = {m.name: m.result() for m in self.metrics[:-1]}
        if 'confusion_matrix_metric' in self.metrics_names:
            self.metrics[-1].fill_output(output)
        return output


    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        y_pred = self(x, training=False)  # Forward pass
        # Compute the loss value.
        # The loss function is configured in `compile()`.
        loss = self.compiled_loss(
            y,
            y_pred,
            regularization_losses=self.losses,
        )

        self.compiled_metrics.update_state(y, y_pred)
        output = {m.name: m.result() for m in self.metrics[:-1]}
        if 'confusion_matrix_metric' in self.metrics_names:
            self.metrics[-1].fill_output(output)
        return output
