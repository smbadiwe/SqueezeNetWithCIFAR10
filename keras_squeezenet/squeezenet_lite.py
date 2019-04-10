from keras_applications.imagenet_utils import _obtain_input_shape
from keras.layers import concatenate,Input, BatchNormalization, Convolution2D, MaxPooling2D, Activation, GlobalAveragePooling2D, Dense, Lambda
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.engine.topology import get_source_inputs
import os

# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import pickle
from funcy import concat, partial, repeat, take
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model, save_model, load_model
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from operator import getitem
import tensorflow as tf
from keras import backend as K


pjoin = os.path.join
sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"


# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = BatchNormalization()(left)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = BatchNormalization()(right)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


# Original SqueezeNet from paper.

def SqueezeNet(include_top=True,
               input_tensor=None, input_shape=None,
               classes=10):
    """Instantiates the SqueezeNet architecture.
    """
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(96, (3, 3), padding='same', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    # x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(2, 2), name='pool4')(x)

    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = MaxPooling2D(pool_size=(2, 2), name='pool8')(x)

    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    x = BatchNormalization()(x)
    # x = Dropout(0.5, name='drop9')(x)
    # x = Convolution2D(1000, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_10')(x)
    x = GlobalAveragePooling2D(name="avgpool10")(x)
    x = Dense(classes, activation='softmax', name="softmax-10")(x)
    # x = Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='squeezenet')

    return model


def euclidean_distance(inputs):
    assert len(inputs) == 2, 'Euclidean distance needs 2 inputs, {} given'.format(len(inputs))
    u, v = inputs
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))  # (?, 1)


def contrastive_loss(y_true, y_pred):
    margin = 1.
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return K.mean(K.equal(y_true, K.cast(K.abs(y_pred) >= 0.5, y_true.dtype)))


def run():
    batch_size = 32
    num_classes = 10
    epochs = 200

    with tf.device("/cpu:0"):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # Convert class vectors to binary class matrices.
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        x_train = x_train.astype('float32', copy=False)
        x_test = x_test.astype('float32', copy=False)
        x_train /= 255
        x_test /= 255

    optimizer = Adam(lr=0.001)

    # input_shape = Input(shape=x_train.shape[1:])
    squeezenet_model_file = './sqz_log/model.h5'
    if os.path.exists(squeezenet_model_file):
        model = load_model(squeezenet_model_file)
    else:
        # train a new SqueezeNet
        model = SqueezeNet(classes=num_classes)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        train_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                        width_shift_range=0.125, height_shift_range=0.125, horizontal_flip=True)
        validation_data = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

        for data in (train_data, validation_data):
            data.fit(x_train)

        callbacks = [
            LearningRateScheduler(partial(getitem, tuple(take(epochs, concat(
                repeat(0.01, 1), repeat(0.1, 99), repeat(0.01, 50), repeat(0.001)))))),
            ModelCheckpoint(filepath=squeezenet_model_file),
            TensorBoard(log_dir="./sqz_log", batch_size=batch_size)
        ]
        results = model.fit_generator(train_data.flow(x_train, y_train, batch_size=batch_size),
                                      steps_per_epoch=x_train.shape[0] // batch_size,
                                      epochs=epochs,
                                      callbacks=callbacks,
                                      validation_data=validation_data.flow(x_test, y_test, batch_size=batch_size),
                                      validation_steps=x_test.shape[0] // batch_size)

        with open('./sqz_log/history.pickle', 'wb') as f:
            pickle.dump(results.history, f)
        save_model(model, squeezenet_model_file)

    # Build the siamese architecture
    model.layers.pop()
    model_cut = Model(name="sqzn_no_softmax", inputs=model.input, outputs=model.layers[-1].output)
    model_cut.load_weights(squeezenet_model_file, by_name=True)
    # with tf.device("/cpu:0"):
    #     model_cut.summary()

    input_shape = x_train.shape[1:]

    im_in1 = Input(shape=input_shape)
    im_in2 = Input(shape=input_shape)
    feat_x1 = model_cut(im_in1)
    feat_x2 = model_cut(im_in2)
    lambda_merge = Lambda(euclidean_distance, output_shape=(1,))([feat_x1, feat_x2])

    siamese = Model(name="siamese", inputs=[im_in1, im_in2], outputs=lambda_merge)
    with tf.device("/cpu:0"):
        siamese.summary()

    optimizer = SGD(momentum=0.9)
    siamese.compile(optimizer=optimizer, loss=contrastive_loss, metrics=[accuracy])

    def make_img_pair(identical, from_train):
        """Select the image pairs"""
        label = np.random.randint(0, num_classes)
        if identical:
            if from_train:
                idx = np.nonzero(y_train[:, label] == 1)[0]
            else:
                idx = np.nonzero(y_test[:, label] == 1)[0]

            # pick any two indexes randomly
            id1 = np.random.randint(0, idx.shape[0])
            id2 = np.random.randint(0, idx.shape[0])
            while id1 == id2:
                id2 = np.random.randint(0, idx.shape[0])
        else:
            if from_train:
                idx1 = np.nonzero(y_train[:, label] == 1)[0]
                idx2 = np.nonzero(y_train[:, (label + 1) % num_classes] == 1)[0]
            else:
                idx1 = np.nonzero(y_test[:, label] == 1)[0]
                idx2 = np.nonzero(y_train[:, (label + 1) % num_classes] == 1)[0]

            # pick any two indexes randomly
            id1 = np.random.randint(0, idx1.shape[0])
            id2 = np.random.randint(0, idx2.shape[0])

        if from_train:
            return np.array([x_train[id1], x_train[id2]])
        else:
            return np.array([x_test[id1], x_test[id2]])

    def generator(from_train):
        while True:
            X = [[None, None]] * batch_size
            y = [[None]] * batch_size
            indexes = np.arange(batch_size)
            identical = True
            for i in indexes:
                X[i] = make_img_pair(identical, from_train)
                y[i] = [1 if identical else 0]
                identical = not identical
            np.random.shuffle(indexes)
            X = np.asarray(X)[indexes]
            y = np.asarray(y)[indexes]
            # print("generator: from_train:", from_train, " - X:", X.shape, "- y:", y.shape)
            yield [X[:, 0], X[:, 1]], y

    siamese_model_file = "./siam_log/siamese.h5"
    epochs = 100
    callbacks = [
        LearningRateScheduler(partial(getitem, tuple(take(epochs, concat(
            repeat(0.01, 1), repeat(0.1, 99), repeat(0.01, 50), repeat(0.001)))))),
        ModelCheckpoint(filepath=siamese_model_file),
        TensorBoard(log_dir="./siam_log", batch_size=batch_size)
    ]
    outputs = siamese.fit_generator(generator(from_train=True), initial_epoch=0,
                                    steps_per_epoch=x_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=generator(from_train=False),
                                    validation_steps=x_test.shape[0] // batch_size, callbacks=callbacks)

    with open('./siam_log/history.pickle', 'wb') as f:
        pickle.dump(outputs.history, f)
    save_model(siamese, siamese_model_file)


if __name__ == "__main__":
    run()
