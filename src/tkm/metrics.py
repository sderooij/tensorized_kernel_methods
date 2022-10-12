import jax.numpy as jnp
from jax import jit

def accuracy(y, y_hat):
    return (jnp.sign(y) == jnp.sign(y_hat)).mean()

def rmse(y,y_hat):
    return jnp.linalg.norm(y-y_hat)

@jit
def loss_function(C,x,y,regularization):
    return jnp.power(jnp.linalg.norm(jnp.dot(C,x) - y),2) + x.T.dot(regularization).dot(x)


@jit
def error(C,x,y):
    return jnp.mean(jnp.sign(C.dot(x)) != y)
