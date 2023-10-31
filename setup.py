from setuptools import setup, find_packages

base_requirements = [
]

cpu_requirements = [
    f'jax[cpu]',
]

gpu_requirements = [
    f'jax[cuda11_cudnn805] @ https://storage.googleapis.com/jax-releases/jax_cuda_releases.html',
]

dev_requirements = [
    f'jupyterlab>=3.0.0',
    f'pandas>=1.2.0',
]

setup(
    name='tkl',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=base_requirements,
    extras_require={
        'dev': dev_requirements,
        'gpu': gpu_requirements,
        'cpu': cpu_requirements,
    },
)
