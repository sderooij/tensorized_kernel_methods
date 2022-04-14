import jax.numpy as jnp


def polynomial(
    X,
    M,
):
    return jnp.power(X[:, None], jnp.arange(M))


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