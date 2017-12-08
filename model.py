from __future__ import print_function
from __future__ import absolute_import

from copy import copy
from builtins import range

import autograd.numpy as np
from autograd.util import flatten
from autograd.core import primitive
import autograd.scipy.stats.norm as norm


def build_gru(input_count, state_count, output_count):
    """Constructor for gated-recurrent unit.

    @param input_count: integer
                        number of input dimensions
    @param state_count: integer
                        number of hidden dimensions
    @param output_count: integer
                         number of binary outputs 
                         (no continuous support at the moment)
    @return predict: function
                     used to predict y_hat
    @return log_likelihood: function
                            used to compute log likelihood
    @return parser: WeightsParser object
                    object to organize weights
    """
    parser = WeightsParser()
    parser.add_shape('init_hiddens', (1, state_count))
    parser.add_shape('update_x_weights', (input_count + 1, state_count))
    parser.add_shape('update_h_weights', (state_count, state_count))
    parser.add_shape('reset_x_weights', (input_count + 1, state_count))
    parser.add_shape('reset_h_weights', (state_count, state_count))
    parser.add_shape('thidden_x_weights', (input_count + 1, state_count))
    parser.add_shape('thidden_h_weights', (state_count, state_count))
    parser.add_shape('output_h_weights', (state_count, output_count))

    def update(curr_input, prev_hiddens, update_x_weights,
               update_h_weights, reset_x_weights, reset_h_weights,
               thidden_x_weights, thidden_h_weights):
        """Update function for GRU."""
        update = sigmoid(np.dot(curr_input, update_x_weights) +
                         np.dot(prev_hiddens, update_h_weights))
        reset = sigmoid(np.dot(curr_input, reset_x_weights) +
                        np.dot(prev_hiddens, reset_h_weights))
        thiddens = np.tanh(np.dot(curr_input, thidden_x_weights) +
                           np.dot(reset * prev_hiddens, thidden_h_weights))
        hiddens = (1 - update) * prev_hiddens + update * thiddens
        return hiddens

    def outputs(weights, input_set, fence_set, output_set=None, return_pred_set=False):
        update_x_weights = parser.get(weights, 'update_x_weights')
        update_h_weights = parser.get(weights, 'update_h_weights')
        reset_x_weights = parser.get(weights, 'reset_x_weights')
        reset_h_weights = parser.get(weights, 'reset_h_weights')
        thidden_x_weights = parser.get(weights, 'thidden_x_weights')
        thidden_h_weights = parser.get(weights, 'thidden_h_weights')
        output_h_weights = parser.get(weights, 'output_h_weights')

        data_count = len(fence_set) - 1
        feat_count = input_set.shape[0]

        ll = 0.0
        n_i_track = 0
        fence_base = fence_set[0]
        pred_set = None

        if return_pred_set:
            pred_set = np.zeros((output_count, input_set.shape[1]))

        # loop through sequences and time steps
        for data_iter in range(data_count):
            hiddens = copy(parser.get(weights, 'init_hiddens'))

            fence_post_1 = fence_set[data_iter] - fence_base
            fence_post_2 = fence_set[data_iter + 1] - fence_base
            time_count = fence_post_2 - fence_post_1
            curr_input = input_set[:, fence_post_1:fence_post_2]

            for time_iter in range(time_count):
                hiddens = update(np.expand_dims(np.hstack((curr_input[:, time_iter], 1)), axis=0),
                                 hiddens, update_x_weights, update_h_weights, reset_x_weights,
                                 reset_h_weights, thidden_x_weights, thidden_h_weights)

                if output_set is not None:
                    # subtract a small number so -1
                    out_proba = sigmoid(np.sign(output_set[:, n_i_track] - 1e-3) *
                                        np.dot(hiddens, output_h_weights))
                    out_lproba = safe_log(out_proba)
                    ll += np.sum(out_lproba)
                else:
                    out_proba = sigmoid(np.dot(hiddens, output_h_weights))
                    out_lproba = safe_log(out_proba)

                if return_pred_set:
                    pred_set[:, n_i_track] = out_lproba[0]

                n_i_track += 1

        return ll, pred_set

    def predict(weights, input_set, fence_set):
        _, output_set = outputs(weights, input_set, fence_set, return_pred_set=True)
        return np.exp(output_set)

    def log_likelihood(weights, input_set, fence_set, output_set):
        ll, _ = outputs(weights, input_set, fence_set, output_set=output_set)
        return ll

    return predict, log_likelihood, parser


def build_mlp(layer_sizes, activation=np.tanh, output_activation=lambda x: x):
    """Constructor for multilayer perceptron.

    @param layer_sizes: list of integers
                        list of layer sizes in the perceptron.
    @param activation: function (default: np.tanh)
                       what activation to use after first N - 1 layers.
    @param output_activation: function (default: linear)
                              what activation to use after last layer.
    @return predict: function
                     used to predict y_hat
    @return log_likelihood: function
                            used to compute log likelihood
    @return parser: WeightsParser object
                    object to organize weights
    """
    parser = WeightsParser()
    for i, shape in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        parser.add_shape(('weights', i), shape)
        parser.add_shape(('biases', i), (1, shape[1]))

    def predict(weights, X):
        cur_X = copy(X.T)
        for layer in range(len(layer_sizes) - 1):
            cur_W = parser.get(weights, ('weights', layer))
            cur_B = parser.get(weights, ('biases', layer))
            cur_Z = np.dot(cur_X, cur_W) + cur_B
            cur_X = activation(cur_Z)
        return output_activation(cur_Z.T)

    def log_likelihood(weights, X, y):
        y_hat = predict(weights, X)
        return mse(y.T, y_hat.T)

    return predict, log_likelihood, parser


def mse(y_true, y_pred):
    return np.mean(np.power(y_true.ravel() - y_pred.ravel(), 2))


def sigmoid(x):
    return 0.5 * (np.tanh(x) + 1)


@primitive
def softplus(x):
    """ Numerically stable transform from real line to positive reals
    Returns np.log(1.0 + np.exp(x))
    Autograd friendly and fully vectorized
    
    @param x: array of values in (-\infty, +\infty)
    @return ans : array of values in (0, +\infty), same size as x
    """
    if not isinstance(x, float):
        mask1 = x > 0
        mask0 = np.logical_not(mask1)
        out = np.zeros_like(x)
        out[mask0] = np.log1p(np.exp(x[mask0]))
        out[mask1] = x[mask1] + np.log1p(np.exp(-x[mask1]))
        return out
    if x > 0:
        return x + np.log1p(np.exp(-x))
    else:
        return np.log1p(np.exp(x))


def make_grad_softplus(ans, x):
    x = np.asarray(x)
    def gradient_product(g):
        return np.full(x.shape, g) * np.exp(x - ans)
    return gradient_product


softplus.defgrad(make_grad_softplus)


def safe_log(x, minval=1e-100):
    return np.log(np.maximum(x, minval))


class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)
