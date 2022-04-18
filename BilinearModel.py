# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 00:28:01 2018
@author: Administrator
"""
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, Input, Reshape, Lambda
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from MyModel import CustomModel

def sign_sqrt(x):
    return K.sign(x) * K.sqrt(K.abs(x) + 1e-10)


def l2_norm(x):
    return K.l2_normalize(x, axis=-1)


def batch_dot(cnn_ab):
    return K.batch_dot(cnn_ab[0], cnn_ab[1], axes=[1, 1])

class BilinearModel():
    def __init__(self, inputs):
        self.inputs = inputs
        self.CreateMyModel()

    def CreateMyModel(self):
        data_augmentation = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomFlip()
        ])
        preprocess_input = keras.applications.vgg16.preprocess_input
        x = data_augmentation(self.inputs)
        x = preprocess_input(x)
        model_vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=x)

        cnn_out_a = model_vgg16.layers[-2].output
        cnn_out_shape = model_vgg16.layers[-2].output_shape
        cnn_out_a = Reshape([cnn_out_shape[1] * cnn_out_shape[2],
                             cnn_out_shape[-1]])(cnn_out_a)
        cnn_out_b = cnn_out_a
        cnn_out_dot = Lambda(batch_dot)([cnn_out_a, cnn_out_b])
        cnn_out_dot = Reshape([cnn_out_shape[-1] * cnn_out_shape[-1]])(cnn_out_dot)

        sign_sqrt_out = Lambda(sign_sqrt)(cnn_out_dot)
        l2_norm_out = Lambda(l2_norm)(sign_sqrt_out)
        output = Dense(64, activation='softmax', name='my_output')(l2_norm_out)

        model = CustomModel(self.inputs, output)
        return model


