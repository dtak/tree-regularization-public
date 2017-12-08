"""Run this file to apply trained GRU-Tree on test set."""

from __future__ import print_function
from __future__ import absolute_import

import cPickle

import autograd.numpy as np
import autograd.numpy.random as npr

from train import GRU
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":
    with open('./trained_models/trained_weights.pkl', 'rb') as fp:
        weights = cPickle.load(fp)['gru']

    gru = GRU(14, 20, 1)
    gru.weights = weights

    with open('./data/test.pkl', 'rb') as fp:
        data_test = cPickle.load(fp)
        X_test = data_test['X']
        F_test = data_test['F']
        y_test = data_test['y']

    y_hat = gru.pred_fun(gru.weights, X_test, F_test)
    auc_test = roc_auc_score(y_test.T, y_hat.T)
    print('Test AUC: {:.2f}'.format(auc_test))
