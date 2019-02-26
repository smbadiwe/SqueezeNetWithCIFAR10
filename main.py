import matplotlib.pyplot as plt
import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..'))) # To import keras_squeezenet.
from keras_squeezenet import SqueezeNet

import tensorflow as tf
from tensorflow import keras
import numpy as np


class KerasLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        # print a dot to show progress
        if batch % 100 == 0:
            print('Epoch: {}'.format(batch))
        print('>', end='')


class KerasCifar10:

    def __init__(self, learning_rate=0.001, validation_split=0.2, batch_size=32, epochs=50,
                 log_dir=None):
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_dir = log_dir

    def get_data(self):
        # Load training and eval data
        print("Load training and eval data")
        (train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar10.load_data()

        # # do not use tf.cast
        # train_data = np.asarray(train_data, dtype=np.float32)
        # train_labels = np.asarray(train_labels, dtype=np.int32)
        # test_data = np.asarray(test_data, dtype=np.float32)
        # test_labels = np.asarray(test_labels, dtype=np.int32)

        print("test_labels Shape after padding: {}".format(test_labels.shape))
        print("train_labels Shape after padding: {}".format(train_labels.shape))
        print("test_data Shape after padding: {}".format(test_data.shape))
        print("train_data Shape after padding: {}".format(train_data.shape))
        tf.summary.image("input_train", train_data)
        tf.summary.image("input_test", test_data)

        # # one-hot encoding for the labels
        # train_labels = keras.utils.to_categorical(train_labels)
        # test_labels = keras.utils.to_categorical(test_labels)
        # train_data /= 255.0
        # test_data /= 255.0

        return (train_data, train_labels), (test_data, test_labels)

    def execute_model(self, qn):
        # Get data
        with tf.device('/cpu:0'):
            (train_data, train_labels), (eval_data, eval_labels) = self.get_data()

        # Build model
        model = SqueezeNet(input_shape=train_data[0].shape, weights=None, classes=10)
        # model = get_model(train_data[0].shape)
        # model.summary()

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        # Train model
        if self.log_dir is not None:
            checkpoint_folder = self.log_dir + "/" + qn
        else:
            checkpoint_folder = "./" + qn

        last_epoch_ran = 0
        from os import path, makedirs
        print("****Checkpoint folder:", checkpoint_folder)
        checkpoint_path = checkpoint_folder + "/train.ckpt"
        if path.exists(checkpoint_path):
            print("\tRestoring checkpoint from %s" % checkpoint_path)
            model.load_weights(checkpoint_path)
            with open(path.join(checkpoint_folder, "epoch.txt"), "r") as f:
                last_epoch_ran = f.read()
            print("\tInitial epoch: %d" % last_epoch_ran)
            last_epoch_ran = int(last_epoch_ran)
        else:
            print("****Creating folder", checkpoint_folder)
            makedirs(checkpoint_folder, exist_ok=True)
        # checkpoint_path = checkpoint_folder + "/train-{epoch:04d}.ckpt"

        class SaveCheckpoint(keras.callbacks.ModelCheckpoint):
            def __init__(self,
                         filepath,
                         monitor='val_loss',
                         verbose=0,
                         save_best_only=False,
                         save_weights_only=False,
                         mode='auto',
                         period=1):
                super(SaveCheckpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                                                     save_best_only=save_best_only,
                                                     save_weights_only=save_weights_only, mode=mode, period=period)

            def on_epoch_end(self, epoch, logs=None):
                super(SaveCheckpoint, self).on_epoch_end(epoch, logs)
                with open(path.join(path.dirname(self.filepath), "epoch.txt"), "w") as f:
                    f.write(str(epoch))

        save_checkpoint = SaveCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        callbacks = [save_checkpoint]

        history = model.fit(train_data, train_labels,
                            batch_size=self.batch_size,
                            epochs=self.epochs, initial_epoch=last_epoch_ran,
                            verbose=1, shuffle=True,
                            validation_split=self.validation_split,
                            callbacks=callbacks)
        # history = model.fit(train_data, train_labels, epochs=self.epochs)
        # Test model
        test_loss, test_acc = model.evaluate(eval_data, eval_labels, batch_size=self.batch_size, verbose=1)

        print("test_loss: {}. test_acc: {}".format(test_loss, test_acc))

        # confusion matrix
        preds = model.predict(eval_data, batch_size=self.batch_size, verbose=1)

        print("eval_labels: {}. preds: {}".format(eval_labels.shape, preds.shape))
        with keras.backend.get_session() as sess:
            conf_mat = tf.confusion_matrix(eval_labels, preds)
            conf_mat = sess.run(conf_mat)

        # clear memory
        keras.backend.clear_session()

        return history, conf_mat, test_loss, test_acc

    @staticmethod
    def plot_loss(epochs, history):
        plt.subplot(2, 2, 1)
        plt.plot(history.history['val_acc'])
        plt.xticks(np.arange(0, epochs, (epochs / 10)))
        plt.title('Val accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')

        plt.subplot(2, 2, 2)
        plt.plot(history.history['val_loss'])
        plt.xticks(np.arange(0, epochs, (epochs / 10)))
        plt.title('Val loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')

        plt.subplot(2, 2, 3)
        plt.plot(history.history['acc'])
        plt.xticks(np.arange(0, epochs, (epochs / 10)))
        plt.title('Train accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')

        plt.subplot(2, 2, 4)
        plt.plot(history.history['loss'])
        plt.xticks(np.arange(0, epochs, (epochs / 10)))
        plt.title('Train loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')

        plt.tight_layout()
        plt.savefig("./squeezenet.png", )
        plt.show()


if __name__ == "__main__":
    k = KerasCifar10()
    history, conf_mat, test_loss, test_acc = k.execute_model("qn1")
    print("Test Acc: {}. Test Loss: {}".format(test_acc, test_loss))
    k.plot_loss(history)
