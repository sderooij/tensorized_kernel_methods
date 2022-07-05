# https://github.com/google/jax/issues/6340#issuecomment-1049973571
FROM nvidia/cuda:11.1-cudnn8-runtime
# 11.6.0-devel-ubuntu20.04

# declare the image name
ENV IMG_NAME=11.1-cudnn8-runtime \
    # declare what jaxlib tag to use
    # if a CI/CD system is expected to pass in these arguments
    # the dockerfile should be modified accordingly
    JAXLIB_VERSION=0.3.14

# RUN nvcc --version

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# install python3-pip
# RUN apt update && apt install python3-pip -y
RUN apt update && apt install -y --no-install-recommends \
    git build-essential \
    python3-dev python3-pip python3-setuptools

# install dependencies via pip
RUN pip3 install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip3 install numpy scipy six wheel jaxlib==${JAXLIB_VERSION}+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_releases.html jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html

WORKDIR /opt/
COPY . .
# COPY setup.py .

RUN pip3 install .[dev]


# FROM nvidia/cuda:10.2-runtime
# WORKDIR /
# RUN apt update && apt install -y --no-install-recommends \
#     git build-essential \
#     python3-dev python3-pip python3-setuptools
# RUN pip3 -q install pip --upgrade
# RUN pip3 install .[dev]