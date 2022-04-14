from functools import partial
import jax.numpy as jnp
from jax import vmap, jit


# @partial(jit, static_argnums=(1,))
# @jit
def polynomial(
    X,
    M,
):
    return jnp.power(X[:, None], jnp.arange(M))


def polynomial_(
    X,
    ar_M,
):
    return jnp.power(X[:, None], ar_M)


def polynomial_vmap(
    X,
    rangeM,
):
    return vmap(jnp.power, (None,0), (-1))(X, rangeM)


def compile_feature_map(
    *args,
    **kwargs,
):
    return jit(partial(polynomial, *args, **kwargs))


def fourier(
    X,
    M,
    lengthscale,
): 
    """
    function Mati = features(X,M,lengthscale)   # fourier features, but can be any polynomials is easiest
    X = (X+1/2)/2;
    w = 1:M;
    S = sqrt(2*pi)*lengthscale*exp(-(pi*w/2).^2*lengthscale^2/2);
    Mati = sinpi(X*w).*sqrt(S);
    end
    """
    ...