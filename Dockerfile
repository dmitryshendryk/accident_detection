FROM  tensorflow/tensorflow:1.13.1-gpu-py3
MAINTAINER Dmitry

ADD . /accident_detection

ARG VID_PATH=None
ENV VID_PATH ${VID_PATH}

ARG GPU_DEVICE=1
ENV GPU_DEVICE ${GPU_DEVICE}

ARG STREAM_TYPE=camera
ENV STREAM_TYPE ${STREAM_TYPE}

ARG WEIGHTS=weights/mask_rcnn_accident_0282_v1.h5
ENV WEIGHTS ${WEIGHTS}

ARG RESPONSE_DELAY=2
ENV RESPONSE_DELAY ${RESPONSE_DELAY}

ARG ACCIDENT_THRESHOLD=70
ENV ACCIDENT_THRESHOLD ${ACCIDENT_THRESHOLD}

###########################
### TENSORFLOW INSTALL  ###
###########################

ARG https_proxy
ARG http_proxy


RUN  apt-get install apt-transport-https
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

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
    qttools5-dev \
    unixodbc-bin \
    unixodbc 

COPY . ./accident_detection



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

## ODBC Drivers

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - 
RUN curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list 

RUN apt-get update  && \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17 && \
    apt-get install -y unixodbc-dev




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
    opencv-python \
    scikit-learn \ 
    matplotlib \
    pandas \
    IPython \ 
    numpy \ 
    pyodbc \
    pypika \
    requests

#WORKDIR /accident_detection/

RUN chmod a+x /accident_detection/install/download_models.sh
RUN chmod a+x /accident_detection/run.sh

# RUN ./accident_detection/run.sh


WORKDIR /accident_detection/workspace/


# && ["sh", "-c", "python3 main.py detect --device=${GPU_DEVICE} --accident_threshold=${ACCIDENT_THRESHOLD}  --streaming=${STREAM_TYPE} --weights=${WEIGHTS} --vid_path=${VID_PATH} --response_delay=${RESPONSE_DELAY}"]
CMD ./../run.sh && python3 main.py detect --device=${GPU_DEVICE} --accident_threshold=${ACCIDENT_THRESHOLD}  --streaming=${STREAM_TYPE} --weights=${WEIGHTS} --vid_path=${VID_PATH} --response_delay=${RESPONSE_DELAY}