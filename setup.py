from setuptools import setup, find_packages

base_requirements = [
    'torch',
    'jax',
]

dev_requirements = [
    'jupyterlab',
]

setup(
    name='tkl',
    version='0.0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=base_requirements,
    extras_require={
        'dev': dev_requirements,
    },
)
