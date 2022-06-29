# tensorized_kernel_methods
Efficient implementation of tensorized kernel methods for different hardware.


# Installation Guide

<!--
## PyTorch
`pip install torch`
-->

## Jax
https://github.com/google/jax/blob/main/README.md#pip-installation-gpu-cuda

`pip install --upgrade "jax[cpu]"`

`pip install jax` && pip install jaxlib

`pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` --> !!! WARNING: jax 0.3.14 does not provide the extra 'cuda11_cudnn811' !!!

### Profiling
https://jax.readthedocs.io/en/latest/profiling.html?highlight=gpu#gpu-profiling

## TKR
`pip install -e ".[dev]"`


## Docker

Build: `docker build -t jtsch/tkm . `

Build and Push: `docker build -t jtsch/tkm . && docker push jtsch/hmsa-seg`

Run with terminal: `docker run -it jtsch/tkm`
