# -*- coding: utf-8 -*-
# 三个256全连接层，进行标准化，对应models:VGG16-256-standard，数据集：Alldatasets

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
from MyCallback import MyCallback

if __name__ == "__main__":
    # tensorflow版本
    print("tf.version:", tf.__version__)

    # 数据集获取
    TRAIN_BATCH_SIZE = 48
    VALID_BATCH_SIZE = 32
    IMG_SIZE = (256, 256)
    path = r"D:\MyFiles\ResearchSubject\Alldatasets"
    train_dataset = image_dataset_from_directory(path, batch_size=TRAIN_BATCH_SIZE, image_size=IMG_SIZE, shuffle=True,
                                                 seed=12, validation_split=0.1, subset='training')
    valid_dataset = image_dataset_from_directory(path, batch_size=VALID_BATCH_SIZE, image_size=IMG_SIZE, shuffle=True,
                                                 seed=12, validation_split=0.1, subset='validation')
    className = train_dataset.class_names  # 这里标签可以这样得到
    for i in range(len(className)):
        c = re.split("_", className[i])
        className[i] = c[1]+"_"+c[2]
    print("64个类：", className)


    def standardize(image_data):
        mean, var = tf.nn.moments(image_data, axes=[0, 1, 2])
        image_data = (image_data - mean) / tf.math.sqrt(var)
        return image_data


    # train_dataset = train_dataset.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    # valid_dataset = valid_dataset.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    train_dataset = train_dataset.map(lambda x, y: (standardize(x), y))
    valid_dataset = valid_dataset.map(lambda x, y: (standardize(x), y))

    # for imgs, labels in train_dataset.take(1):
    #     print(labels)

    # def displayImages(dataset):
    #     plt.figure(figsize=(10, 10))
    #     # 整个画布（包括各子图在内）的大小是1000×1000
    #     for imags, labels in dataset.take(1):
    #         # 取一个batch的数据
    #         for i in range(9):
    #             # img = tf.squeeze(imags[i], 2)
    #             plt.subplot(3, 3, i + 1)
    #             plt.imshow(imags[i])
    #             plt.title(class_names[labels[i]])
    #             plt.axis('off')
    #     plt.show()
    #
    # displayImages(train_dataset)
    # displayImages(valid_dataset)

    # 提前取好数据
    AUTOTUNE = tf.data.AUTOTUNE
    train_batch_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    valid_batch_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

    # 创建模型
    inputs = keras.Input(shape=(256, 256, 3), name="my_input")
    from MyModel2 import MyModel2
    model = MyModel2(inputs).CreateMyModel()

    # 加载上次结束训练时的权重
    # model.load_weights(r"D:\MyFiles\ResearchSubject\door4\doorModels/VGG16-256-standard\cp-023-1.581-0.463-0.557-0.450-1.907-0.385-0.667-0.355.ckpt")
    # print('successfully loading weights')
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["acc", TopkAccuracy(2),
                                                                                               ConfusionMatrixMetric(64)])
    EPOCHS = 100
    INITIAL_EPOCH = 23

    myCallback = MyCallback()
    cp_callback = myCallback.CheckPointCallback(path=r'D:\MyFiles\ResearchSubject\door4\doorModels/VGG16-256-standard')
    tensorboard_callback = myCallback.TensorboardCallback(log_dir=r'D:\MyFiles\ResearchSubject\door4\doorTensorboard/VGG16-256-standard')
    learningrate_callback = myCallback.LearningRateCallback()
    model.fit(train_batch_dataset, epochs=EPOCHS+INITIAL_EPOCH, initial_epoch=INITIAL_EPOCH, validation_data=valid_batch_dataset,
              callbacks=[cp_callback, tensorboard_callback])
