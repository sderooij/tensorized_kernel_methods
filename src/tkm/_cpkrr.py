"""
Implementation of functions for the CP-KRR model.
Functions are defined here to keep estimator classes clean.

In CPKRR we solve the follwing
min_w[d] <g(x_d), w_d> + reg

reg = l * <w_d, w_d>
g(x_d) = phi(x_d) ox ()
"""
import jax
import jax.numpy as jnp
from jax.lax import fori_loop


# @partial(jit, static_argnames=['W'])
def _init_reg(d, reg, W):  # TODO: forloop is not necessary, should be able to do this with linalg
    """
    Computes the regularization term for the d-th factor matrix of the weight tensor.

    Args:
        d (int): The factor matrix of the weight tensor to compute the regularization term for.
        reg (float): The regularization strength.
        W (jax.interpreters.xla.DeviceArray): The weight tensor.

    Returns:
        jax.interpreters.xla.DeviceArray: The regularization term for the d-th factor matrix of the weight tensor.
    """
    reg *= jnp.dot(W[d].T, W[d])  # reg has shape R * R
    return reg


# @partial(jit, static_argnames=['X', 'W'])
def _init_g(d, G, feature_func, X, W):
    """
    Initializes the G matrix for a given dimension d.
    min_w[d] <G(x_d), w_d> + reg

    Args:
        feature_func (Callable): The feature function to use.
        d (int): The dimension for which to initialize G.
        G (jax.numpy.ndarray): The G matrix to initialize.
        X (jax.numpy.ndarray): The input data matrix of shape (N, D).
        W (jax.numpy.ndarray): The weight matrix of shape (D, R).

    Returns:
        jax.numpy.ndarray: The initialized Matd matrix of shape (N, R).
    """
    phi_x = feature_func(X[:, d])
    G *= jnp.dot(phi_x, W[d])  # G has shape N * R, contraction of phi_x_d * w_d
    return G