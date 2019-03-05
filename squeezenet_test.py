import matplotlib.pyplot as plot
import pickle

from funcy    import last, partial
from operator import getitem


def plot_history(history):
    legends = ["train loss", "test loss", "train accuracy", "test accuracy"]
    i = 0

    def plot_values_collection(title, values_collection):
        plot.clf()
        plot.title(title)
        for values in values_collection:
            plot.plot(values, label=legends.pop(0))
        plot.legend()
        plot.ylabel(title.split(' ')[0])
        plot.xlabel("Epochs")
        plot.show()

    plot_values_collection('Loss',     map(partial(getitem, history), ('loss', 'val_loss')))
    plot_values_collection('Accuracy', map(partial(getitem, history), ('acc',  'val_acc')))


def main():
    with open('./results/history.pickle', 'rb') as f:
        history = pickle.load(f)

    print(last(history['acc']))
    print(last(history['val_acc']))
    print(last(history['loss']))
    print(last(history['val_loss']))

    plot_history(history)


if __name__ == '__main__':
    main()
