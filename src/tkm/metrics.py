import jax.numpy as jnp

def accuracy(y, y_hat):
    return (jnp.sign(y) == jnp.sign(y_hat)).mean()

def rmse(y,y_hat):
    return jnp.linalg.norm(y-y_hat)
