# FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
FROM  tensorflow/tensorflow:1.13.1-gpu-py3
MAINTAINER Dmitry

ADD . /accident_detection

###########################
### TENSORFLOW INSTALL  ###
###########################

ARG https_proxy
ARG http_proxy

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    curl  \
    qt5-default \
    qtbase5-dev \
    qttools5-dev 

RUN apt-get install -y \
    libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    libjasper-dev libavformat-dev libpq-dev libxine2-dev libglew-dev \
    libtiff5-dev zlib1g-dev libjpeg-dev libpng12-dev libjasper-dev \
    libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev \
	libswscale-dev libeigen3-dev libtbb-dev libgtk2.0-dev 
    # libcudnn7=7.1.4.18-1+cuda9.0 

RUN apt-get install -y \
    python3-dev \
    python3-numpy \
    python3-pip 

## Cleanup
RUN rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip3 --no-cache-dir install \
    numpy \
    hdf5storage \
    h5py \
    scipy \
    py3nvml \ 
    keras==2.2.0 \ 
    scikit-image==0.15.0 \
    opencv-python

#ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /accident_detection


