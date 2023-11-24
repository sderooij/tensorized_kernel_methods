import jax.numpy as jnp
from jax import jit
from jax import vmap
from functools import partial


# @jit
def dotkron(a, b):
    """
    Row-wise kronecker product

    For column wise inspiration check khatri_rao: 
    https://scipy.github.io/devdocs/reference/generated/scipy.linalg.khatri_rao.html
    """
    # column-wise python
    # np.vstack([np.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T

    # row-wise matlab
    # y = repmat(L,1,c2) .* kron(R, ones(1, c1));
    # TODO check if transpose of repmat for second term option is: repmat() .* repmat()' 

    # TODO jax improvement for loops
    # TODO does python provide efficient kron() for vectors
    return jnp.vstack([jnp.kron(b[k, :], a[k,:]) for k in jnp.arange(b.shape[0])])  # shift a and b due to big-endian ordering


@jit
def vmap_dotkron(a,b):
    return vmap(jnp.kron)(b, a) # shift a and b due to big-endian ordering


def vmap_dotkron_contracted(a, b, y): # TODO: check y
    temp = vmap_dotkron(a,b)
    return temp.T @ temp, temp.T @ y


def batched_dotkron(A,B,y,batch_size=10000, **kwargs):
    N,DA = A.shape
    _,DB = B.shape
    CC = jnp.zeros((DA*DB,DA*DB))
    Cy = jnp.zeros((DA*DB,1))
    for n in range(0,N,batch_size): # TODO fori jax loop
        idx = min(n+batch_size,N)
        temp = vmap_dotkron(A[n:idx,:], B[n:idx,:]) # repmat(A(n:idx,:),1,DB)*kron(B(n:idx,:), ones(1, DA))
        CC += temp.T @ temp
        Cy += temp.T @ y[n:idx,:]
    return CC, Cy


def get_dotkron(batch_size=None):
    if batch_size is None:
        return vmap_dotkron_contracted # TODO: should have the right return types
    else:
        return partial(batched_dotkron, batch_size=batch_size)
