"""Run this file generate a toy dataset."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np


def gen_synthetic_dataset(data_count, time_count):
    """Signal-and-Noise HMM dataset

    The generative process comes from two separate HMM processes. First, 
    a "signal" HMM generates the first 7 data dimensions from 5 well-separated states. 
    Second, an independent "noise" HMM generates the remaining 7 data dimensions 
    from a different set of 5 states. Each timestep's output label is produced by a 
    rule involving both the signal data and the signal hidden state.

    @param data_count: number of sequences in dataset
    @param time_count: number of timesteps in a sequence
    @return obs_set: Torch Tensor data_count x time_count x 14
    @return out_set: Torch Tensor data_count x time_count x 1
    """

    bias_mat = np.array([15])
    # 5 states + 7 observations
    weight_mat = np.array([[10, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0]])
    
    state_count = 5 
    dim_count = 7
    out_count = 1

    # signal HMM process
    pi_mat_signal = np.array([.5, .5, 0, 0, 0])
    trans_mat_signal = np.array(([.7, .3, 0, 0, 0],
                                 [.5, .25, .25, 0, 0],
                                 [0, .25, .5, .25, 0],
                                 [0, 0, .25, .25, .5],
                                 [0, 0, 0, .5, .5]))
    obs_mat_signal = np.array(([.5, .5, .5, .5, 0, 0, 0],
                               [.5, .5, .5, .5, .5, 0, 0],
                               [.5, .5, .5, 0, .5, 0, 0],
                               [.5, .5, .5, 0, 0, .5, 0],
                               [.5, .5, .5, 0, 0, 0, .5]))

    # noise HMM process
    pi_mat_noise = np.array([.2, .2, .2, .2, .2])
    trans_mat_noise = np.array(([.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2],
                                [.2, .2, .2, .2, .2]))
    obs_mat_noise = np.array(([.5, .5, .5, 0, 0, 0, 0],
                              [0, .5, .5, .5, 0, 0, 0],
                              [0, 0, .5, .5, .5, 0, 0],
                              [0, 0, 0, .5, .5, .5, 0],
                              [0, 0, 0, 0, .5, .5, .5]))

    # create the sequences
    obs_set = np.zeros((dim_count * 2, time_count, data_count))
    out_set = np.zeros((out_count, time_count, data_count))   
    
    state_set_signal = np.zeros((state_count, time_count, data_count))
    state_set_noise = np.zeros((state_count, time_count, data_count))
    
    # loop through to sample HMM states
    for data_ix in range(data_count):
        for time_ix in range(time_count):
            if time_ix == 0:
                state_signal = np.random.multinomial(1, pi_mat_signal)
                state_noise = np.random.multinomial(1, pi_mat_noise)
                state_set_signal[:, 0, data_ix] = state_signal
                state_set_noise[:, 0, data_ix] = state_noise
            else:
                tvec_signal = np.dot(state_set_signal[:, time_ix - 1, data_ix], trans_mat_signal)
                tvec_noise = np.dot(state_set_noise[:, time_ix - 1, data_ix], trans_mat_noise)
                state_signal = np.random.multinomial(1, tvec_signal)
                state_noise = np.random.multinomial(1, tvec_noise)
                state_set_signal[:, time_ix, data_ix] = state_signal
                state_set_noise[:, time_ix, data_ix] = state_noise
    
    # loop through to generate observations and outputs
    for data_ix in range(data_count):
        for time_ix in range(time_count):
            obs_vec_signal = np.dot(state_set_signal[:, time_ix, data_ix], obs_mat_signal)
            obs_vec_noise = np.dot(state_set_noise[:, time_ix, data_ix], obs_mat_noise)
            obs_signal = np.random.binomial(1, obs_vec_signal)
            obs_noise = np.random.binomial(1, obs_vec_noise)
            obs = np.hstack((obs_signal, obs_noise))  # concat together
            obs_set[:, time_ix, data_ix] = obs
            
            # input is state concatenated with observation
            in_vec = np.hstack((state_set_signal[:, time_ix, data_ix],
                                obs_set[:dim_count, time_ix, data_ix]))
            
            # output is a logistic regression on W \dot input
            out_vec = 1 / (1 + np.exp(-1 * (np.dot(weight_mat, in_vec) - bias_mat)))
            
            out = np.random.binomial(1, out_vec)
            out_set[:, time_ix, data_ix] = out

    return obs_set, out_set


if __name__ == "__main__":
    import os
    import cPickle

    def map_3d_to_2d(X_arr, y_arr):
        """Convert 3d NumPy array to 2d NumPy array with fenceposts.

        @param X_arr: NumPy array
                      size D x T x N 
        @param t_arr: NumPy array
                      size O x T x N 
        @return X_arr_DM: NumPy array
                          size D x (T x N)
        @return y_arr_DM: NumPy array
                          size O x (T x N)
        @return fenceposts_Np1: NumPy array (1-dimensional)
                                represents splits between sequences.
        """
        n_in_dims, n_timesteps, n_seqs = X_arr.shape
        n_out_dims, _, _ = y_arr.shape

        X_arr_DM = X_arr.swapaxes(0, 2).reshape((-1, n_in_dims)).T
        y_arr_DM = y_arr.swapaxes(0, 2).reshape((-1, n_out_dims)).T

        fenceposts_Np1 = np.arange(0, (n_seqs + 1) * n_timesteps, n_timesteps)
        return X_arr_DM, fenceposts_Np1, y_arr_DM

    obs_set, out_set = gen_synthetic_dataset(20, 10)
    obs_train, out_train = obs_set[:, :, :10], out_set[:, :, :10]
    obs_test, out_test = obs_set[:, :, 10:], out_set[:, :, 10:]

    obs_train, fcpt_train, out_train = map_3d_to_2d(obs_train, out_train)
    obs_test, fcpt_test, out_test = map_3d_to_2d(obs_test, out_test)

    if not os.path.isdir('./data'):
        os.mkdir('./data')

    with open('./data/train.pkl', 'wb') as fp:
        cPickle.dump({'X': obs_train, 'F': fcpt_train, 'y': out_train}, fp)

    with open('./data/test.pkl', 'wb') as fp:
        cPickle.dump({'X': obs_test, 'F': fcpt_test, 'y': out_test}, fp) 
