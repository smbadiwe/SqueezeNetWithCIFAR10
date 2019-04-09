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
from data_set import load_data
from funcy import concat, identity, juxt, partial, rcompose, repeat, take
from keras.callbacks import LearningRateScheduler
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, \
    GlobalAveragePooling2D, Input, Lambda, MaxPooling2D
from keras.models import Model, save_model, load_model
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import plot_model
from operator import getitem
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras import backend as K


def computational_graph(class_size):
    # Utility functions.

    def ljuxt(*fs):  # Kerasはジェネレーターを引数に取るのを嫌がるみたい、かつ、funcyはPython3だと積極的にジェネレーターを使うみたいなので、リストを返すjuxtを作りました。
        return rcompose(juxt(*fs), list)

    def batch_normalization():
        return BatchNormalization()

    def relu():
        return Activation('relu')

    def conv(filters, kernel_size):
        return Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(
            0.0001))  # ReLUするならウェイトをHe初期化するのが基本らしい。あと、Kerasにはweight decayがなかったのでkernel_regularizerで代替したのたけど、これで正しい？

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


def generate_confusion_matrix(model, x_validation, y_validation, batch_size):
    # Confution Matrix and Classification Report
    Y_pred = model.predict(x_validation, batch_size=batch_size)
    y_pred = np.argmax(Y_pred, axis=1)
    y_valid = np.argmax(y_validation, axis=1)

    print('Confusion Matrix')
    cm = (confusion_matrix(y_valid, y_pred))
    print(cm)
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    colorbar=True,
                                    show_absolute=False,
                                    show_normed=True)
    plt.show()
    del model


def euclidean_distance(inputs):
    assert len(inputs) == 2, 'Euclidean distance needs 2 inputs, {} given'.format(len(inputs))
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred):
    margin = 1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))


def siamese(im_in, squeezenet_model):  # modelsqueeze): output of squeezenet model
    # squeezenet_model = Dense(512, activation="relu")(squeezenet_model)
    # squeezenet_model = Dropout(0.2)(squeezenet_model)
    last_layer = squeezenet_model.layers[-1]
    # print("last_layer: name:", last_layer.name, "shape:", last_layer.output.shape)
    # d1 = Dense(128, activation="linear")
    # d1.build((None, 512))  # , int(last_layer.output.shape[1])))=
    # squeezenet_model.layers.append(d1)
    # l1 = Lambda(lambda x: K.l2_normalize(x, axis=1))
    # l1.build((None, 512))  # , int(last_layer.output.shape[1])))
    # squeezenet_model.layers.append(l1)
    squeezenet_model.summary()
    # return None

    model_top = squeezenet_model
    # feat_x = Dense(128, activation="linear")(last_layer.output)
    # feat_x = Lambda(lambda x: K.l2_normalize(x, axis=1))(feat_x)
    model_top = Model(inputs=[im_in], outputs=last_layer.output)
    model_top.summary()

    print("im_in shape:", im_in.shape, "squeezenet_model type:", type(squeezenet_model))
    im_in1 = Input(shape=im_in.shape)
    im_in2 = Input(shape=im_in.shape)

    feat_x1 = model_top(im_in1)
    feat_x2 = model_top(im_in2)

    lambda_merge = Lambda(euclidean_distance)([feat_x1, feat_x2])

    model_final = Model(inputs=[im_in1, im_in2], outputs=lambda_merge)
    model_final.summary()

    adam = Adam(lr=0.001)

    # sgd = SGD(lr=0.001, momentum=0.9)

    model_final.compile(optimizer=adam, loss=contrastive_loss)


def main():
    import os
    with tf.device("/cpu:0"):
        (x_train, y_train), (x_validation, y_validation) = load_data()

    batch_size = 32
    epochs = 200
    input_shape = Input(shape=x_train.shape[1:])
    model_file = './results/model.h5'
    if os.path.exists(model_file):
        model = load_model(model_file)
        # with tf.device("/cpu:0"):
        #     validation_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    else:
        model = Model(*juxt(identity, computational_graph(y_train.shape[1]))(input_shape))
        model.compile(loss='categorical_crossentropy', optimizer=SGD(momentum=0.9), metrics=['accuracy'])

        with tf.device("/cpu:0"):
            train_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                            width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
            validation_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

        for data in (train_data, validation_data):
            data.fit(x_train)  # 実用を考えると、x_validationでのfeaturewiseのfitは無理だと思う……。

        results = model.fit_generator(train_data.flow(x_train, y_train, batch_size=batch_size),
                                      steps_per_epoch=x_train.shape[0] // batch_size,
                                      epochs=epochs,
                                      callbacks=[LearningRateScheduler(partial(getitem, tuple(take(epochs, concat(
                                          repeat(0.01, 1), repeat(0.1, 99), repeat(0.01, 50), repeat(0.001))))))],
                                      validation_data=validation_data.flow(x_validation, y_validation,
                                                                           batch_size=batch_size),
                                      validation_steps=x_validation.shape[0] // batch_size)

        with open('./results/history.pickle', 'wb') as f:
            pickle.dump(results.history, f)
        save_model(model, model_file)

    try:
        with tf.device("/cpu:0"):
            # model.summary()
            # print("=== AFTER POPPING THE LAST ===")
            model.layers.pop()
            # model.summary()
            # generate_confusion_matrix(model, x_validation, y_validation, batch_size)
            # plot_model(model, to_file='./results/model.png')
    except Exception as ex:
        print("plot_model failed with error:", repr(ex), "\nMoving on...")

    siamese(input_shape, model)


if __name__ == '__main__':
    main()
