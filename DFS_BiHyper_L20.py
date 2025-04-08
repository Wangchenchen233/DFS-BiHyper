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


def compute_W_i_l20_ori(p, alpha):
    sorted_indices = np.argsort(np.abs(p))[::-1]
    sorted_p = p[sorted_indices]

    if alpha < sorted_p[-1] ** 2:
        w = p
    elif alpha > sorted_p[0] ** 2:
        w = np.zeros_like(p)
    else:
        for k in range(1, len(sorted_p) + 1):
            if sorted_p[k - 1] ** 2 < alpha < sorted_p[k - 2] ** 2:
                w = np.zeros_like(p)
                w[sorted_indices[:k - 1]] = sorted_p[:k - 1]
                break
    return w

def compute_W_i_l20(p, alpha):
    abs_square = np.abs(p) ** 2
    sorted_indices = np.argsort(abs_square)[::-1]
    sorted_abs_square = abs_square[sorted_indices]

    if alpha < sorted_abs_square[-1]:
        return p

    elif alpha > sorted_abs_square[0]:
        return np.zeros_like(p)
    else:
        k = np.searchsorted(sorted_abs_square, alpha, side='right')
        w = np.zeros_like(p)
        w[sorted_indices[:k]] = p[sorted_indices[:k]]
        return w


def DFS_BiHyper_L20(X, Y, num_k, verbose=True):
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
    W_all = []
    row_norms_history = []

    for iter_step in range(max_iter):
        # update d^i, Y^-i
        for i in range(X.shape[1]):
            Y_i = compute_Y_i(Y, X, W, i)
            d_i = X[:, i].T @ Y_i / a_i_all[i]
            d_i_all[i, :] = d_i
            W[i, :] = compute_W_i_l20(d_i_all[i, :], para_a)

        W_all.append(W.copy())

        W_num.append(np.sum(np.any(W != 0, axis=1)))
        largest_elements = np.max(np.abs(d_i_all), axis=1)
        sorted_indices = np.argsort(largest_elements)[::-1]
        sorted_largest_elements = largest_elements[sorted_indices]
        sorted_largest_elements_squared = sorted_largest_elements ** 2
        para_a = (sorted_largest_elements_squared[num_k - 1] + sorted_largest_elements_squared[
            num_k]) / 2

        paras.append(para_a)
        obj.append(np.linalg.norm(X @ W - Y) ** 2 + para_a * calculate_l21_norm(W))
        obj2.append(np.linalg.norm(X @ W - Y))
        row_norms = np.linalg.norm(W, axis=1)
        row_norms_history.append(row_norms)
        if verbose:
            print('obj at iter {0}: with obj: {1:.2f} with para_a: {2:.2f}'.format(iter_step + 1, obj[iter_step],
                                                                                   paras[iter_step]))
            if iter_step >= 20 and math.fabs(obj[iter_step] - obj[iter_step - 1]) < 1e-4:
                break

    return W, paras, obj, obj2, W_num
