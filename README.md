# tensorized_kernel_methods
Efficient implementation of tensorized kernel methods for different hardware.


# Installation Guide

<!--
## PyTorch
`pip install torch`
-->
`pip install --upgrade pip`

`pip install wheel`

## Jax

### CPU
`pip install --upgrade "jax[cpu]"`

<!--`pip install jax` && pip install jaxlib-->

### GPU
https://github.com/google/jax/blob/main/README.md#pip-installation-gpu-cuda

`pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html` --> !!! WARNING: jax 0.3.14 does not provide the extra 'cuda11_cudnn811' !!!

### Mac
Jax on apple silicon: https://developer.apple.com/metal/jax/



## TKR
`pip install -e ".[dev]"`


# Profiling with JAX
https://jax.readthedocs.io/en/latest/profiling.html?highlight=gpu#gpu-profiling




# Docker

Build: `docker build -t jtsch/tkm . `

Build and Push: `docker build -t jtsch/tkm . && docker push jtsch/tkm`

Pull: `docker pull jtsch/tkm`

Run with terminal: `docker run -it jtsch/tkm`
