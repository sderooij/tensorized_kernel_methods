from functools import partial
import jax.numpy as jnp
from jax import vmap, jit


# @partial(jit, static_argnums=(1,))
# @jit
def polynomial(
    X,
    M,
    *args,
    **kwargs,
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
    feature_map,
    *args,
    **kwargs,
):
    # return jit(
    return partial(feature_map, *args, **kwargs)
        # )


def fourier(
    X,
    M,
    lengthscale,
    *args,
    **kwargs,
): 
    """
    function Mati = features(X,M,lengthscale)   # fourier features, but can be any polynomials is easiest
    X = (X+1/2)/2;
    w = 1:M;
    S = sqrt(2*pi) * lengthscale*exp(-(pi*w/2).^2*lengthscale^2/2);
    Mati = sinpi(X*w).*sqrt(S);
    end
    """
    X = (X+1/2)/2
    w = jnp.arange(1,M+1)
    S = jnp.sqrt(2*jnp.pi)*lengthscale * jnp.exp(- jnp.power((jnp.pi*w/2),2) * jnp.power(lengthscale,2) /2)
    return jnp.sin(jnp.pi*jnp.outer(X,w)) * jnp.sqrt(S)
