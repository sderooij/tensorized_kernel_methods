import jax.numpy as jnp

def polynomial(
    X,
    M,
    lengthscale,
):
    return jnp.power(X, list(range(M)))

def fourier(): 
    """
    function Mati = features(X,M,lengthscale)   # fourier features, but can be any polynomials is easiest
    X = (X+1/2)/2;
    w = 1:M;
    S = sqrt(2*pi)*lengthscale*exp(-(pi*w/2).^2*lengthscale^2/2);
    Mati = sinpi(X*w).*sqrt(S);
    end
    """
    ...