from setuptools import setup, find_packages

base_requirements = [
    'jax[cpu]',
]

gpu_requirements = [
    'jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
]

dev_requirements = [
    'jupyterlab',
    'pandas',
]

setup(
    name='tkl',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=base_requirements,
    extras_require={
        'dev': dev_requirements,
        'gpu': dev_requirements + gpu_requirements,
    },
)
