# -*- coding: utf-8 -*-
from matplotlib.ticker import FuncFormatter
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np

from DFS_BiHyper_L20 import DFS_BiHyper_L20
from DFS_BiHyper_L21 import DFS_BiHyper_L21


def format_func(value, tick_number):
    return f'{value:.1f}' if abs(value) >= 0.1 else f'{value:.1f}'


if __name__ == '__main__':
    iris = load_breast_cancer()  # Load the Iris dataset

    X = iris.data
    X = preprocessing.scale(X)
    y = iris.target.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    Y = encoder.fit_transform(y)

    fig, axs = plt.subplots(3, 5, figsize=(20, 8))
    for idx, num_k in enumerate(np.arange(2, 11, 2)):
        # W, paras, obj, obj2, W_num = DFS_BiHyper_L20(X, Y, num_k, verbose=False)
        W, paras, obj, obj2, W_num = DFS_BiHyper_L21(X, Y, num_k, verbose=False)

        axs[0, idx].plot(paras, color='blue')
        axs[0, idx].set_title(f'$k$ = {num_k}')
        if idx == 0:
            axs[0, idx].set_ylabel(r'Hyperparameter $\alpha$')
        axs[0, idx].ticklabel_format(axis="y", style="sci", scilimits=(-1, 1))

        axs[1, idx].plot(obj, color='green')
        if idx == 0:
            axs[1, idx].set_ylabel('Objective value')
        axs[1, idx].yaxis.set_major_formatter(FuncFormatter(format_func))
        axs[0, idx].yaxis.get_offset_text().set_visible(False)

        axs[2, idx].plot(W_num, color='red')
        axs[2, idx].set_xlabel('Iterations')
        axs[2, idx].set_ylim(0, 12)
        if idx == 0:
            axs[2, idx].set_ylabel('Non-zero rows in $W$')

    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.show()
