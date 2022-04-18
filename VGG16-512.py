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

TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# tensorflow版本
print("tf.version:", tf.__version__)

# 数据集获取
num_skipped = 0
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
#
AUTOTUNE = tf.data.AUTOTUNE
train_batch_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
valid_batch_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip(),
    # # 默认为水平或者竖直反转
    # keras.layers.experimental.preprocessing.RandomRotation(0.2),
    # keras.layers.experimental.preprocessing.RandomContrast(0.1)
])

preprocess_input = keras.applications.vgg16.preprocess_input
#
# plt.figure(figsize=(10, 10))
# for images, labels in train_dataset.take(1):
#     # 取了1个batch_size的数据
#     # print(images, labels)
#     print(images[0])
#     first_image = images[0]
#     print(preprocess_input(images[0]))  # 显示预处理对图片做了什么，将RGB→BGR，且去均值
#     print(np.max(preprocess_input(images[0])), np.min(preprocess_input(images[0])))
#     for i in range(20):
#         ax = plt.subplot(4, 5, i+1)
#         augmented_image = data_augmentation(tf.expand_dims(first_image, 0))  # 数据增强的输入必须是4维
#         plt.imshow(augmented_image[0].numpy().astype('uint8'))
#         plt.axis('off')
# plt.show()

# keras.backend.set_learning_phase(0)
base_model = keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg')
# keras.backend.set_learning_phase(1)
base_model.trainable = False

fc1 = keras.layers.Dense(512, activation='relu', name='dense_1')
drop1 = keras.layers.Dropout(0.5, name='dropout_1')
fc2 = keras.layers.Dense(512, activation='relu', name='dense_2')
drop2 = keras.layers.Dropout(0.5, name='dropout_2')
fc3 = keras.layers.Dense(64, activation='softmax', name='output')

inputs = keras.Input(shape=(224, 224, 3), name="input_1")

x = preprocess_input(inputs)
x = data_augmentation(x)
x = base_model(x)
x = fc1(x)
x = drop1(x)
x = fc2(x)
x = drop2(x)
outputs = fc3(x)

class MyModel(keras.Model):
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

model = MyModel(inputs, outputs)

model.load_weights(r"D:\MyFiles\ResearchSubject\door4\doorModels\VGG16-ckpt-512\avg\cp-314-0.633-0.782-0.920-0.387-0.894-0.965.ckpt")
print('successfully loading weights')
model.summary()

# def MyAccuracy(y_true, y_pred, k):
#     a = tf.nn.in_top_k(y_true, y_pred, k)
#     MyAccuracy = tf.math.count_nonzero(a) / tf.size(a)
#     return MyAccuracy
#
class TopkAccuracy(keras.metrics.Metric):
    def __init__(self, k=2, name='top-2_acc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.k = k
        self.topk = self.add_weight(name="topk", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values, indices = tf.math.top_k(y_pred, k=self.k, sorted=True)
        y = tf.reshape(y_true, [-1, 1])
        correct = tf.cast(tf.equal(y, indices), tf.float32)
        top_k_accuracy = tf.reduce_mean(correct) * self.k
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
        precision = tf.reduce_mean(precision)
        recall = tf.reduce_mean(recall)
        f1 = tf.reduce_mean(f1)

        return precision, recall, f1

    def fill_output(self, output):
        results = self.result()
        output['precision'] = results[0]
        output['recall'] = results[1]
        output['F1'] = results[2]

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=[keras.metrics.SparseCategoricalAccuracy(),
                                                                                           TopkAccuracy(2), ConfusionMatrixMetric(64)])
EPOCHS = 100
INITIAL_EPOCH = 314
# loss0, accuracy0 = model.evaluate(validation_dataset)
# print('未开始训练前，预训练模型的效果：initial loss:{:.2f}'.format(loss0))
# print('initial acc:{:.2f}'.format(accuracy0))

log_dir = r'D:\MyFiles\ResearchSubject\door4\doorTensorboard/VGG16-ckpt-512/avg'
path = r'D:\MyFiles\ResearchSubject\door4\doorModels/VGG16-ckpt-512/avg'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(path, exist_ok=True)
CheckPoint_Path = path + '/cp-{epoch:03d}-{loss:.3f}-{sparse_categorical_accuracy:.3f}-{acc:.3f}-{val_loss:.3f}-' \
                         '{val_sparse_categorical_accuracy:.3f}-{val_acc:.3f}.ckpt'
cp_callback = keras.callbacks.ModelCheckpoint(CheckPoint_Path, verbose=2, save_weights_only=True)

def decay(epoch):
    step_size = 1600
    iterations = epoch*41
    base_lr = 1e-06
    max_lr = 3.65e-05
    cycle = np.floor(1 + iterations / (2 * step_size))
    x = np.abs(iterations / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * x
    return lr

callback1 = tf.keras.callbacks.LearningRateScheduler(decay)
# cp_callback = keras.callbacks.ModelCheckpoint(CheckPoint_Path, monitor='val_loss', verbose=2, save_best_only=True,
#                                               save_weights_only=True, mode='min')
tensorboard_callback = keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
# cp_callback = keras.callbacks.ModelCheckpoint(hdf5_Path, verbose=2, save_best_only=True, save_weights_only=True)

# model.evaluate(train_batch_dataset)
model.fit(train_batch_dataset, epochs=EPOCHS+INITIAL_EPOCH, initial_epoch=INITIAL_EPOCH, validation_data=valid_batch_dataset,
          callbacks=[cp_callback, tensorboard_callback, callback1])

# plt.figure(figsize=(10, 10))
# for images, labels in train_batch_dataset.take(1):
#     predictList = model.predict(images)
#     predictList = np.argmax(predictList, axis=1)
#     print(predictList)
#     print(labels)
#     # 取了1个batch_size的数据
#     # print(images, labels)
#     for i in range(63):
#         if i % 9 == 0:
#             plt.figure(figsize=(10, 10))
#             p = 0
#         plt.subplot(3, 3, p + 1)
#         plt.imshow(images[i].numpy().astype('uint8'))
#         if className[labels[i]] != className[predictList[i]]:
#             plt.title(className[labels[i]] + '/' + className[predictList[i]], color='red')
#         else:
#             plt.title(className[labels[i]] + '/' + className[predictList[i]])
#         plt.axis('off')
#         p += 1
#     plt.show()
