from keras_applications.imagenet_utils import _obtain_input_shape
from keras.layers import concatenate,Input, BatchNormalization, Convolution2D, MaxPooling2D, Activation, GlobalAveragePooling2D, Dense, Lambda
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.engine.topology import get_source_inputs
import os
import glob

# import numpy as np
# import matplotlib.pyplot as plt
# from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import pickle
from funcy import concat, partial, repeat, take
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model, save_model, load_model
from keras.optimizers import Adam
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
    return K.sqrt(K.sum((K.square(u - v)), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred):
    margin = 1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))


def dd_make_img_pair(fpaths, identical=True):
    folder1 = folder2 = np.random.choice(fpaths)
    if not identical:
        while os.path.samefile(folder1, folder2):
            folder2 = np.random.choice(fpaths)
    rgb_path1 = rgb_path2 = np.random.choice(glob(pjoin(folder1, "*.bmp")))
    while os.path.samefile(rgb_path1, rgb_path2):
        rgb_path2 = np.random.choice(glob(pjoin(folder2, "*.bmp")))
    depth_path1 = rgb_path1[:-5] + "d.dat"
    depth_path2 = rgb_path2[:-5] + "d.dat"

    # process image 1
    rgb1 = Image.open(rgb_path1)
    rgb1.thumbnail((640, 480))
    rgb1 = np.asarray(rgb1)[140:340, 220:420, :3]
    depth1 = pd.read_csv(depth_path1, sep='\t', header=None)
    depth1[(depth1 > 3000) | (depth1 == -1)] = 3000
    depth1 = depth1.values[140:340, 220:420]
    depth1 = (depth1 - np.mean(depth1)) / np.max(depth1)
    rgbd1 = np.dstack((rgb1, depth1))

    # process image 2
    rgb2 = Image.open(rgb_path1)
    rgb2.thumbnail((640, 480))
    rgb2 = np.asarray(rgb2)[140:340, 220:420, :3]
    depth2 = pd.read_csv(depth_path2, sep='\t', header=None)
    depth2[(depth2 > 3000) | (depth2 == -1)] = 3000
    depth2 = depth2.values[140:340, 220:420]
    depth2 = (depth2 - np.mean(depth2)) / np.max(depth2)
    rgbd2 = np.dstack((rgb2, depth2))

    return np.array([rgbd1, rgbd2])


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

    with tf.device("/cpu:0"):
        model.summary()

    # Build the siamese architecture
    model.layers.pop()
    input_shape = x_train.shape[1:]
    im_in1 = Input(shape=input_shape)
    im_in2 = Input(shape=input_shape)
    feat_x1 = model(im_in1)
    feat_x2 = model(im_in2)

    lambda_merge = Lambda(euclidean_distance, output_shape=input_shape)([feat_x1, feat_x2])

    siamese = Model(input=[im_in1, im_in2], output=lambda_merge, name="siamese")
    siamese.summary()

    siamese.compile(optimizer=optimizer, loss=contrastive_loss)

    def make_img_pair(identical, from_train):
        # TODO: Select the image pairs
        if identical:
            if from_train:
                pass
            else:
                pass
        else:
            if from_train:
                pass
            else:
                pass

    def generator(from_train):
        while True:
            X = []
            y = []
            identical = True
            for _ in range(batch_size):
                X.append(make_img_pair(identical, from_train))
                y.append(np.array([0.]))
                identical = not identical
            X = np.asarray(X)
            y = np.asarray(y)
            yield [X[:, 0], X[:, 1]], y

    callbacks = [
        LearningRateScheduler(partial(getitem, tuple(take(epochs, concat(
            repeat(0.01, 1), repeat(0.1, 99), repeat(0.01, 50), repeat(0.001)))))),
        ModelCheckpoint(filepath=squeezenet_model_file),
        TensorBoard(log_dir="./siam_log", batch_size=batch_size)
    ]
    outputs = siamese.fit_generator(generator(from_train=True),
                                    steps_per_epoch=x_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=generator(from_train=False),
                                    validation_steps=x_test.shape[0] // batch_size, callbacks=callbacks)

    with open('./siam_log/history.pickle', 'wb') as f:
        pickle.dump(outputs.history, f)
    save_model(siamese, './siam_log/siamese.h5')


if __name__ == "__main__":
    run()
