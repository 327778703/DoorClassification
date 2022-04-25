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


if __name__ == "__main__":
    # tensorflow版本
    print("tf.version:", tf.__version__)

    # 数据集获取
    TRAIN_BATCH_SIZE = 64
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

    # 提前取好数据
    AUTOTUNE = tf.data.AUTOTUNE
    train_batch_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    valid_batch_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

    # 创建模型
    inputs = keras.Input(shape=(224, 224, 3), name="input_1")
    model = MyModel(inputs).CreateMyModel()

    # 加载上次结束训练时的权重
    model.load_weights(r"D:\MyFiles\ResearchSubject\door4\doorModels\VGG16-ckpt-512\avg\cp-325-0.525-0.838-0.921-0.838-0.565-0.833-1.000-0.830.ckpt")
    print('successfully loading weights')
    model.summary()
    cm = ConfusionMatrixMetric(64)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=[keras.metrics.SparseCategoricalAccuracy(),
                                                                                               TopkAccuracy(2), cm])
    model.evaluate(train_batch_dataset)
    plt.figure(figsize=(10, 10))
    for images, labels in train_batch_dataset.take(1):
        predictList = model.predict(images)
        predictList = np.argmax(predictList, axis=1)
        print(predictList)
        print(labels)
        # 取了1个batch_size的数据
        # print(images, labels)
        for i in range(63):
            if i % 9 == 0:
                plt.figure(figsize=(10, 10))
                p = 0
            plt.subplot(3, 3, p + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            if className[labels[i]] != className[predictList[i]]:
                plt.title(className[labels[i]] + '/' + className[predictList[i]], color='red')
            else:
                plt.title(className[labels[i]] + '/' + className[predictList[i]])
            plt.axis('off')
            p += 1
        plt.show()
    # while True:
    #     try:
    #         images, labels = next(iter(train_batch_dataset))
    #         predictList = model.predict(images)
    #         _ = cm.update_state(labels, predictList)
    #     except StopIteration:
    #         break
    fig, ax = plt.subplots(figsize=(15, 15))
    cm_matrix = cm.total_cm.numpy().astype(np.int32)
    ax.matshow(cm_matrix, cmap=plt.cm.Reds)
    for i in range(64):
        for j in range(64):
            c = cm_matrix[j, i]
            ax.text(i, j, str(c), va='center', ha='center')
    plt.xticks(range(64), ['{} {}'.format(className[i], i) for i in range(64)], rotation=270)
    plt.yticks(range(64), ['{} {}'.format(className[i], i) for i in range(64)])
    plt.show()
