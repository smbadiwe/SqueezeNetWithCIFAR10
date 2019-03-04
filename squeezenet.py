# Implementation of SqueezeNet
#
# SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size
# https://arxiv.org/abs/1602.07360
#
# Identity Mappings in Deep Residual Networks
# https://arxiv.org/abs/1603.05027)# Wide Residual Network

from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import pickle
import matplotlib.pyplot as plt
from data_set                  import load_data
from funcy                     import concat, identity, juxt, partial, rcompose, repeat, take
from keras.callbacks           import LearningRateScheduler
from keras.layers              import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input, MaxPooling2D
from keras.models              import Model, save_model, load_model
from keras.optimizers          import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers        import l2
from keras.utils               import plot_model
from operator                  import getitem
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf


def computational_graph(class_size):
    # Utility functions.

    def ljuxt(*fs):  # Kerasはジェネレーターを引数に取るのを嫌がるみたい、かつ、funcyはPython3だと積極的にジェネレーターを使うみたいなので、リストを返すjuxtを作りました。
        return rcompose(juxt(*fs), list)

    def batch_normalization():
        return BatchNormalization()

    def relu():
        return Activation('relu')

    def conv(filters, kernel_size):
        return Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))  # ReLUするならウェイトをHe初期化するのが基本らしい。あと、Kerasにはweight decayがなかったのでkernel_regularizerで代替したのたけど、これで正しい？

    def concatenate():
        return Concatenate()

    def add():
        return Add()

    def max_pooling():
        return MaxPooling2D()

    def dropout():
        return Dropout(0.5)

    def global_average_pooling():
        return GlobalAveragePooling2D()

    def dense(units, activation):
        return Dense(units, activation=activation, kernel_regularizer=l2(0.0001))

    # Define SqueezeNet.

    def fire_module(filters_squeeze, filters_expand):
        return rcompose(batch_normalization(),
                        relu(),
                        conv(filters_squeeze, 1),
                        batch_normalization(),
                        relu(),
                        ljuxt(conv(filters_expand // 2, 1),
                              conv(filters_expand // 2, 3)),
                        concatenate())

    def fire_module_with_shortcut(filters_squeeze, filters_expand):
        return rcompose(ljuxt(fire_module(filters_squeeze, filters_expand),
                              identity),
                        add())

    return rcompose(conv(96, 3),
                    # max_pooling(),
                    fire_module(16, 128),
                    fire_module_with_shortcut(16, 128),
                    fire_module(32, 256),
                    max_pooling(),
                    fire_module_with_shortcut(32, 256),
                    fire_module(48, 384),
                    fire_module_with_shortcut(48, 384),
                    fire_module(64, 512),
                    max_pooling(),
                    fire_module_with_shortcut(64, 512),
                    batch_normalization(),
                    relu(),
                    global_average_pooling(),
                    dense(class_size, 'softmax'))


def main():
    with tf.device("/cpu:0"):
        (x_train, y_train), (x_validation, y_validation) = load_data()

    model = Model(*juxt(identity, computational_graph(y_train.shape[1]))(Input(shape=x_train.shape[1:])))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.9), metrics=['accuracy'])

    model.summary()
    try:
        with tf.device("/cpu:0"):
            plot_model(model, to_file='./results/model.png')
    except Exception as ex:
        print("plot_model failed with error:", repr(ex), "\nMoving on...")

    with tf.device("/cpu:0"):
        train_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
        validation_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    for data in (train_data, validation_data):
        data.fit(x_train)  # 実用を考えると、x_validationでのfeaturewiseのfitは無理だと思う……。

    batch_size = 32
    epochs = 250

    # history_file = './results/history.pickle'
    # results_file = './results/model.h5'
    # import os
    # if os.path.exists(history_file):
    #     with open(history_file, 'wb') as f:
    #         history = pickle.load(f)
    # else:
    results = model.fit_generator(train_data.flow(x_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  callbacks=[LearningRateScheduler(partial(getitem, tuple(take(epochs, concat(repeat(0.01, 1), repeat(0.1, 99), repeat(0.01, 50), repeat(0.001))))))],
                                  validation_data=validation_data.flow(x_validation, y_validation, batch_size=batch_size),
                                  validation_steps=x_validation.shape[0] // batch_size)

    # Confution Matrix and Classification Report
    Y_pred = model.predict_generator(validation_data.flow(x_validation, y_validation, batch_size=batch_size), x_validation.shape[0]// batch_size)
    y_pred = np.argmax(Y_pred, axis=1)
    y_valid = np.argmax(y_validation, axis=1)
    print(np.argmax(y_validation, axis=1))
    print(len(y_pred))

    with open('./results/history.pickle', 'wb') as f:
        pickle.dump(results.history, f)
    save_model(model, './results/model.h5')

    print('Confusion Matrix')
    # plt.imshow((y_valid, y_pred))
    # plt.show()
    cm = (confusion_matrix(y_valid, y_pred))
    print(confusion_matrix(y_valid, y_pred))
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True)
    plt.show()
    del model


if __name__ == '__main__':
    main()
