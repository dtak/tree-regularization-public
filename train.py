"""Run this file to train demo GRU-Tree."""

from __future__ import print_function
from __future__ import absolute_import

import pydotplus
from copy import deepcopy

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.optimizers import sgd, adam
from autograd import grad

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from datasets import gen_synthetic_dataset
from model import build_gru, build_mlp
from model import softplus


class GRUTree(object):
    def __init__(self, in_count, state_count, hidden_sizes, out_count, strength=200.0):
        self.gru = GRU(in_count, state_count, out_count)
        self.mlp = MLP(self.gru.num_weights, hidden_sizes, 1)
        self.mlp.weights = self.mlp.init_weights(0.1)  # init for 1st run
        self.strength = strength

    def objective(self, W, X, F, y):
        path_length = self.mlp.pred_fun(self.mlp.weights, 
                                        W[:self.gru.num_weights][:, np.newaxis]).ravel()[0]
        return -self.gru.loglike_fun(W, X, F, y) + self.strength * path_length

    def average_path_length_batch(self, weights_batch, X_train, F_train, y_train):
        apl_batch = np.zeros((1, weights_batch.shape[1]))
        for i in range(weights_batch.shape[1]):
            apl_batch[0, i] = self.gru.average_path_length(weights_batch[:, i], 
                                                           X_train, F_train, y_train)
        return apl_batch

    def train(self, X_train, F_train, y_train, iters_retrain=25, num_iters=1000,
              batch_size=32, lr=1e-3, param_scale=0.01, log_every=10):
        npr.seed(42)
        num_retrains = num_iters // iters_retrain
        for i in xrange(num_retrains):
            self.gru.objective = self.objective
            # carry over weights from last training
            init_weights = self.gru.weights if i > 0 else None
            print('training deep net... [%d/%d], learning rate: %.4f' % (i + 1, num_retrains, lr))
            self.gru.train(X_train, F_train, y_train, num_iters=iters_retrain,
                           batch_size=batch_size, lr=lr, param_scale=param_scale, 
                           log_every=log_every, init_weights=init_weights)
            print('building surrogate dataset...')
            W_train = deepcopy(self.gru.saved_weights.T)
            APL_train = self.average_path_length_batch(W_train, X_train, F_train, y_train)
            print('training surrogate net... [%d/%d]' % (i + 1, num_retrains))
            self.mlp.train(W_train[:self.gru.num_weights, :], APL_train, num_iters=3000,
                           lr=1e-3, param_scale=0.1, log_every=250)

        self.pred_fun = self.gru.pred_fun
        self.weights = self.gru.weights
        # save final decision tree
        self.tree = self.fit_tree(self.weights, X_train, F_train, y_train)

        return self.weights


class GRU(object):
    def __init__(self, in_count, state_count, out_count):
        super(GRU, self).__init__()
        pred_fun, loglike_fun, parser = build_gru(in_count, state_count, out_count)
        self.num_weights = parser.num_weights
        self.in_count = in_count
        self.out_count = out_count
        self.state_count = state_count
        self.pred_fun = pred_fun
        self.loglike_fun = loglike_fun
        self.saved_weights = None

    def objective(self, W, X, F, y):
        return -self.loglike_fun(W, X, F, y)

    def init_weights(self, param_scale):
        return npr.randn(self.num_weights) * param_scale

    def fit_tree(self, weights, X_train, F_train, y_train):
        """Train decision tree to track path length."""
        y_train_hat = self.pred_fun(weights, X_train, F_train)
        y_train_hat_int = np.rint(y_train_hat).astype(int)
        tree = DecisionTreeClassifier(min_samples_leaf=25)
        tree.fit(X_train.T, y_train_hat_int.T)
        return tree

    def average_path_length(self, weights, X_train, F_train, y_train):
        tree = self.fit_tree(weights, X_train, F_train, y_train)
        path_length = average_path_length(tree, X_train.T)
        return path_length

    def train(self, X_train, F_train, y_train, batch_size=32, num_iters=1000, 
              lr=1e-3, param_scale=0.01, log_every=100, init_weights=None):
        grad_fun = build_batched_grad_fences(grad(self.objective), batch_size, 
                                             X_train, F_train, y_train)
        if init_weights is None:
            init_weights = self.init_weights(param_scale)
        saved_weights = np.zeros((num_iters, self.num_weights))

        def callback(weights, i, gradients):
            apl = self.average_path_length(weights, X_train, F_train, y_train)
            saved_weights[i, :] = weights
            loss_train = self.objective(weights, X_train, F_train, y_train)
            if i % log_every == 0: 
                print('model: gru | iter: {} | loss: {:.2f} | apl: {:.2f}'.format(i, loss_train, apl))

        optimized_weights = adam(grad_fun, init_weights, num_iters=num_iters, 
                                 step_size=lr, callback=callback)
        self.saved_weights = saved_weights
        self.weights = optimized_weights
        return optimized_weights


class MLP(object):
    def __init__(self, in_count, hidden_sizes, out_count):
        super(MLP, self).__init__()
        layer_specs = [in_count] + hidden_sizes + [out_count]
        pred_fun, loglike_fun, parser = build_mlp(layer_specs, output_activation=softplus)
        self.num_weights = parser.num_weights
        self.in_count = in_count
        self.out_count = out_count
        self.hidden_sizes = hidden_sizes
        self.pred_fun = pred_fun
        self.loglike_fun = loglike_fun

    def objective(self, W, X, y):
        return self.loglike_fun(W, X, y) + np.linalg.norm(W[:self.num_weights], ord=2)

    def init_weights(self, param_scale):
        return npr.randn(self.num_weights) * param_scale

    def train(self, X_train, y_train, batch_size=32, num_iters=1000, 
              lr=1e-3, param_scale=0.01, log_every=100, init_weights=None):
        grad_fun = build_batched_grad(grad(self.objective), batch_size, 
                                      X_train, y_train)
        if init_weights is None:
            init_weights = self.init_weights(param_scale)
    
        def callback(weights, i, gradients):
            loss_train = self.objective(weights, X_train, y_train)
            if i % log_every == 0: 
                print('model: mlp | iter: {} | loss: {:.2f}'.format(i, loss_train))

        optimized_weights = adam(grad_fun, init_weights, num_iters=num_iters,
                                 step_size=lr, callback=callback)
        self.weights = optimized_weights
        return optimized_weights


def average_path_length(tree, X):
    """Compute average path length: cost of simulating the average
    example; this is used in the objective function.

    @param tree: DecisionTreeClassifier instance
    @param X: NumPy array (D x N)
              D := number of dimensions
              N := number of examples
    @return path_length: float
                         average path length
    """
    leaf_indices = tree.apply(X)
    leaf_counts = np.bincount(leaf_indices)
    leaf_i = np.arange(tree.tree_.node_count)
    path_length = np.dot(leaf_i, leaf_counts) / float(X.shape[0])
    return path_length


def get_ith_minibatch_ixs(i, num_data, batch_size):
    """Split data into minibatches.
    
    @param i: integer
              iteration index
    @param num_data: integer
                     number of data points
    @param batch_size: integer
                       number of data points in a batch
    @return batch_slice: slice object
    """
    num_minibatches = num_data / batch_size + ((num_data % batch_size) > 0)
    i = i % num_minibatches
    start = i * batch_size
    stop = start + batch_size
    return slice(start, stop)


def build_batched_grad(grad, batch_size, inputs, targets):
    """Return grad on batched gradient. 

    @param grad: gradient function.
    @param batch_size: integer
                       batch size
    @param inputs: NumPy Array
                   size D x N
    @param targets: NumPy Array
                    size O x N
    @return batched_grad: function
                          function to compute gradients on inputs.
    """
    def batched_grad(weights, i):
        cur_idxs = get_ith_minibatch_ixs(i, targets.shape[1], batch_size)
        return grad(weights, inputs[:, cur_idxs], targets[:, cur_idxs])
    return batched_grad


def get_ith_minibatch_ixs_fences(b_i, batch_size, fences):
    """Split timeseries data of uneven sequence lengths into batches.
    This is how we handle different sized sequences.
    
    @param b_i: integer
                iteration index
    @param batch_size: integer
                       size of batch
    @param fences: list of integers
                   sequence of cutoff array
    @return idx: integer
    @return batch_slice: slice object
    """
    num_data = len(fences) - 1
    num_minibatches = num_data / batch_size + ((num_data % batch_size) > 0)
    b_i = b_i % num_minibatches
    idx = slice(b_i * batch_size, (b_i+1) * batch_size)
    batch_i = np.arange(num_data)[idx]
    batch_slice = np.concatenate([range(i, j) for i, j in 
                                  zip(fences[batch_i], fences[batch_i+1])])
    return idx, batch_slice


def build_batched_grad_fences(grad, batch_size, inputs, fences, targets):
    """Return grad on batched gradient. 

    @param grad: gradient function.
    @param batch_size: integer
                       batch size
    @param inputs: NumPy Array
                   size D x N
    @param fences: NumPy Array
    @param targets: NumPy Array
                    size O x N
    @return batched_grad: function
                          function to compute gradients on inputs with fenceposts.
    """
    def batched_grad(weights, i):
        cur_idxs, cur_slice = get_ith_minibatch_ixs_fences(i, batch_size, fences)
        batched_inputs = inputs[:, cur_slice]
        batched_targets = None if targets is None else targets[:, cur_slice]
        batched_fences = fences[cur_idxs.start:cur_idxs.stop+1] - fences[cur_idxs.start]
        return grad(weights, batched_inputs, batched_fences, batched_targets)
    return batched_grad


def visualize(tree, save_path):
    """Generate PDF of a decision tree.

    @param tree: DecisionTreeClassifier instance
    @param save_path: string 
                      where to save tree PDF
    """
    dot_data = export_graphviz(tree, out_file=None,
                               filled=True, rounded=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph = make_graph_minimal(graph)  # remove extra text

    if not save_path is None:
        graph.write_pdf(save_path)


def make_graph_minimal(graph):
    nodes = graph.get_nodes()
    for node in nodes:
        old_label = node.get_label()
        label = prune_label(old_label)
        if label is not None:
            node.set_label(label)
    return graph


def prune_label(label):
    if label is None:
        return None
    if len(label) == 0:
        return None
    label = label[1:-1]
    parts = [part for part in label.split('\\n')
             if 'gini =' not in part and 'samples =' not in part]
    return '"' + '\\n'.join(parts) + '"'


if __name__ == "__main__":
    import os
    import cPickle

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strength', type=float, default=1000.0,
                        help='how much to weigh tree-regularization term.')
    args = parser.parse_args()

    with open('./data/train.pkl', 'rb') as fp:
        data_train = cPickle.load(fp)
        X_train = data_train['X']
        F_train = data_train['F']
        y_train = data_train['y']

    gru = GRUTree(14, 20, [25], 1, strength=args.strength)
    gru.train(X_train, F_train, y_train, iters_retrain=25, num_iters=300,
              batch_size=10, lr=1e-2, param_scale=0.1, log_every=10)

    if not os.path.isdir('./trained_models'):
        os.mkdir('./trained_models')

    with open('./trained_models/trained_weights.pkl', 'wb') as fp:
        cPickle.dump({'gru': gru.gru.weights, 'mlp': gru.mlp.weights}, fp)
        print('saved trained model to ./trained_models')

    visualize(gru.tree, './trained_models/tree.pdf')
    print('saved final decision tree to ./trained_models')
