import glob
import os

import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix

from base.constants import FILMS_GENRE

"""
Analytics for classification problem
"""


class Analytics:

    @staticmethod
    def confusion_matrix_analysis(Y_shuffled, Y_preds, model_name, epoch):
        plt.clf()
        conf_mat = confusion_matrix(Y_shuffled, Y_preds)

        # Normalise the array
        highest, lowest = np.max(conf_mat), np.min(conf_mat)
        value_range = highest - lowest
        conf_mat = conf_mat * (1.0 / value_range)

        # Plot a heat map
        ax = sns.heatmap(conf_mat, linewidths=0.3, xticklabels=FILMS_GENRE, yticklabels=FILMS_GENRE)
        ax.set(title=f'Confusion at epoch: {epoch}')
        os.makedirs(f"outputs/{model_name}/confusion/", exist_ok=True)
        ax.figure.savefig(f"outputs/{model_name}/confusion/{epoch}.png")
        plt.clf()

    @staticmethod
    def create_animated_gifs(model_name):
        images = [Image.open(image) for image in sorted(glob.glob(f"outputs/{model_name}/confusion/*.png"))]
        images[0].save(f"outputs/{model_name}/confusion/confusion.gif", format="GIF", append_images=images,
                       save_all=True, duration=len(images) / 2, loop=0)

    @staticmethod
    def plot_loss(train_loss, val_loss, model_name):
        Analytics.create_dirs_if_not_present(model_name)
        plt.clf()
        epochs = np.arange(len(train_loss))
        plt.title("Training and Validation Loss")
        plt.plot(epochs, train_loss, label="Train Loss")
        plt.plot(epochs, val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"outputs/{model_name}/loss.png")
        plt.clf()

    @staticmethod
    def plot_acc(train_acc, val_acc, model_name):
        Analytics.create_dirs_if_not_present(model_name)
        plt.clf()
        epochs = np.arange(len(train_acc))
        plt.title("Training and Validation Acc")
        plt.plot(epochs, train_acc, label="Train Acc")
        plt.plot(epochs, val_acc, label="Validation Acc")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"outputs/{model_name}/loss.png")
        plt.clf()

    @staticmethod
    def show_train_val_test_stats(y_train, y_test, y_val):
        ytr, yva, yte = np.array(y_train).astype(int), np.array(y_val).astype(int), np.array(y_test).astype(int)
        print (np.unique(ytr, return_counts=True))
        print (np.unique(yva, return_counts=True))
        print (np.unique(yte, return_counts=True))


    @staticmethod
    def acc_by_class(Y_shuffled, Y_preds, model_name):
        Analytics.create_dirs_if_not_present(model_name)
        truth = np.array(Y_shuffled, dtype=int)
        preds = np.array(Y_preds, dtype=int)

        is_correct = (truth == preds).astype(int)

        accuracies = []

        for cat in range(len(FILMS_GENRE)):
            ind = np.where(truth == cat)
            total = len(ind[0]) + 1e-8
            correct = np.sum(is_correct[ind])

            accuracies.append(correct / total)

        plt.clf()
        plt.title("Accuracies by class")
        plt.barh(FILMS_GENRE, accuracies)
        plt.xlabel("Classes")
        plt.ylabel("Accuracy")
        plt.savefig(f"outputs/{model_name}/acc_by_class.png")
        plt.clf()
        return accuracies

    @staticmethod
    def plot_value_counts(df: pandas.DataFrame, model_name):
        Analytics.create_dirs_if_not_present(model_name)
        plt.clf()
        plt.title("Frequencies by class")
        plt.xlabel("Frequency")
        plt.ylabel("Classes")
        df['genre'].value_counts().plot(kind="barh")
        plt.savefig(f"outputs/{model_name}/value_counts.png", bbox_inches="tight")
        plt.clf()

    @staticmethod
    def create_dirs_if_not_present(model_name):
        os.makedirs(f"outputs/{model_name}", exist_ok=True)

    @staticmethod
    def test_confusion():
        a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        for i in range(10):
            b = [i] * 10
            Analytics.confusion_matrix_analysis(a, b, "test", i)
        Analytics.create_animated_gifs("test")

    @staticmethod
    def test_loss():
        a = [1] * 1000
        b = [2] * 1000
        Analytics.plot_loss(a, b, "test")

    @staticmethod
    def test_value_counts():
        df = pandas.DataFrame(columns=["genre"], data=[["drama"], ["comedy"], ["comedy"]])
        Analytics.plot_value_counts(df, "test")

# Analytics.test_confusion()
# Analytics.test_loss()
# Analytics.acc_by_class([0, 0, 0, 0, 1], [0, 0, 0, 2, 1], "test")
# Analytics.test_value_counts()
