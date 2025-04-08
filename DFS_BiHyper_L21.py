# -*- coding: utf-8 -*-
import numpy as np
from utils.sparse_learning import calculate_l21_norm
import math

def compute_Y_i(Y, X, W, i):
    '''
    :param Y: n,c
    :param X: n,d
    :param W: d,c
    :param i:
    :return:
    '''
    Y_minus_i = Y.copy()
    j_indices = np.arange(X.shape[1]) != i
    Y_minus_i -= X[:, j_indices] @ W[j_indices, :]
    return Y_minus_i


def compute_W_i(d_i, a_i, para_a):
    '''
    :param d_i:
    :param a_i:
    :param para_a:
    :return:
    '''
    norm_d = np.linalg.norm(d_i, 2)
    scaling_factor = max(1 - para_a / (2 * a_i * norm_d), 0)
    W_i = scaling_factor * d_i
    return W_i


def DFS_BiHyper_L21(X, Y, num_k, verbose=True):
    n_samp = X.shape[0]
    H = np.eye(n_samp) - np.ones((n_samp, n_samp)) / n_samp
    X = H @ X
    Y = H @ Y
    a_i_all = np.linalg.norm(X, axis=0) ** 2
    W = np.zeros((X.shape[1], Y.shape[1]))

    para_a = 0
    d_i_all = np.zeros_like(W)
    max_iter = 100
    obj = []
    obj2 = []
    paras = []
    W_num = []

    row_norms_history = []
    for iter_step in range(max_iter):
        # update d^i, Y^-i
        for i in range(X.shape[1]):
            Y_i = compute_Y_i(Y, X, W, i)
            d_i = X[:, i].T @ Y_i / a_i_all[i]
            d_i_all[i, :] = d_i
            W[i, :] = compute_W_i(d_i_all[i, :], a_i_all[i], para_a)

        W_num.append(np.sum(np.any(W != 0, axis=1)))

        # update alpha
        d_weight = 2 * a_i_all * np.sqrt(np.multiply(d_i_all, d_i_all).sum(1))
        d_w_sort = np.sort(d_weight)
        para_a = (d_w_sort[-num_k - 1] + d_w_sort[-num_k]) / 2
        # para_a = d_w_sort[-num_k - 1]

        paras.append(para_a)
        obj.append(np.linalg.norm(X @ W - Y) ** 2 + para_a * calculate_l21_norm(W))
        obj2.append(np.linalg.norm(X @ W - Y))
        row_norms = np.linalg.norm(W, axis=1)
        row_norms_history.append(row_norms)
        if verbose:
            print('obj at iter {0}: with obj: {1:.2f} with para_a: {2:.2f}'.format(iter_step + 1, obj[iter_step],
                                                                                   paras[iter_step]))
            if iter_step >= 1 and math.fabs(obj2[iter_step] - obj2[iter_step - 1]) < 1e-4:
                break
    return W, paras, obj, obj2, W_num

