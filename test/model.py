"""
    Test functions using some precomputed values.
"""

import pytest
import jax
import jax.numpy as jnp
import tkm
from tkm import _cpkrr as cpkrr
from tkm.model import TensorizedKernelMachine as TKM

import pickle


def test_TensorizedKernelMachine():
    """
    Test the TensorizedKernelMachine class.
    """
    with open('test/test_model.pkl', 'rb') as f:
        model_params = pickle.load(f)
    f.close()
    X = model_params['X']
    y = model_params['y']
    W = model_params['W']
    features = model_params['feature_map']
    l = model_params['l']
    M = model_params['M']
    w_init = model_params['W_init']
    num_sweeps = model_params['num_sweeps']
    R = model_params['R']
    model = TKM(features=features, R=R, l=l, num_sweeps=num_sweeps, M=M)
    model = model.fit(X, y)
    assert jnp.allclose(model.W_, W)


if __name__ == '__main__':
    test_TensorizedKernelMachine()