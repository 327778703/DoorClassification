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
from MyCallback import MyCallback
from BilinearModel import BilinearModel
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

if __name__ == "__main__":
    # tensorflow版本
    print("tf.version:", tf.__version__)

    # 数据集获取
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    path = r"D:\MyFiles\ResearchSubject\3dmax\doorModel_datasets"
    train_dataset = image_dataset_from_directory(path, batch_size=TRAIN_BATCH_SIZE, image_size=IMG_SIZE, shuffle=True,
                                                 seed=12, validation_split=0.3, subset='training')
    valid_dataset = image_dataset_from_directory(path, batch_size=VALID_BATCH_SIZE, image_size=IMG_SIZE, shuffle=True,
                                                 seed=12, validation_split=0.3, subset='validation')
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

    # train_dataset = train_dataset.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    # valid_dataset = valid_dataset.map(lambda x, y: (tf.image.rgb_to_grayscale(x), y))
    # print(gray_ds.take(1))


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
    inputs = keras.Input(shape=(224, 224, 3), name="my_input")
    model = BilinearModel(inputs).CreateMyModel()

    # 加载上次结束训练时的权重
    # model.load_weights(r"D:\MyFiles\ResearchSubject\DoorClassification\doorModels\VGG16-ckpt-512\avg\cp-504-1.059-0.662-0.933-0.662-0.608-0.788-1.000-0.788.ckpt")
    # print('successfully loading weights')
    model.summary()
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-6),
                  metrics=["acc", TopkAccuracy(2), ConfusionMatrixMetric(64)])
    EPOCHS = 100
    INITIAL_EPOCH = 0

    myCallback = MyCallback()
    cp_callback = myCallback.CheckPointCallback(path=r'D:\MyFiles\ResearchSubject\door4\doorModels/BilinearCNN-standard')
    tensorboard_callback = myCallback.TensorboardCallback(log_dir=r'D:\MyFiles\ResearchSubject\door4\doorTensorboard/BilinearCNN-standard')
    # learningrate_callback = myCallback.LearningRateCallback()
    learningrate_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_delta=1e-5, patience=2, verbose=1,
                                  min_lr=0.00000001)
    #
    model.fit(train_batch_dataset, epochs=EPOCHS+INITIAL_EPOCH, initial_epoch=INITIAL_EPOCH, validation_data=valid_batch_dataset,
              callbacks=[cp_callback, tensorboard_callback, learningrate_callback])

