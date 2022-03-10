import jax.numpy as jnp

def dotkron(a, b):
    """
    Row-wise kronecker product

    For column wise inspiration check khatri_rao: 
    https://scipy.github.io/devdocs/reference/generated/scipy.linalg.khatri_rao.html
    """
    # column-wise python
    # np.vstack([np.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T

    # row-wise matlab
    # y = repmat(L,1,c2).*kron(R, ones(1, c1));
    # TODO check if transpose of repmat for second term option is: repmat() .* repmat()' 

    # TODO jax improvement for loops
    # TODO does python provide efficient kron() for vectors
    return jnp.vstack([jnp.kron(a[k, :], b[k, :]) for k in range(b.shape[0])])
