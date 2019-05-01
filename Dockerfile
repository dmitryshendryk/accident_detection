FROM  tensorflow/tensorflow:1.13.1-gpu-py3
MAINTAINER Dmitry

ADD . /accident_detection

ARG VID_PATH=videos_accident/cctv_1.mp4
ENV VID_PATH ${VID_PATH}

ARG GPU_DEVICE=1
ENV GPU_DEVICE ${GPU_DEVICE}

ARG STREAM_TYPE=camera
ENV STREAM_TYPE ${STREAM_TYPE}

ARG WEIGHTS=weights/mask_rcnn_accident_0282_v1.h5
ENV WEIGHTS ${WEIGHTS}

ARG RESPONSE_DELAY=2
ENV RESPONSE_DELAY ${RESPONSE_DELAY}

ARG MODELS_SERVER=http://ec2-18-217-76-76.us-east-2.compute.amazonaws.com:8090/file/accident_detection/accident_detection_v_02.h5
ENV MODELS_SERVER ${MODELS_SERVER}

###########################
### TENSORFLOW INSTALL  ###
###########################

ARG https_proxy
ARG http_proxy

######### Download model from sandbox
RUN wget $MODELS_SERVER -O /accident_detection/weights


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

#ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

WORKDIR /accident_detection/workspace/


CMD ["sh", "-c", "python3 main.py detect --device=${GPU_DEVICE}  --streaming=${STREAM_TYPE} --weights=${WEIGHTS} --vid_path=${VID_PATH} --response_delay=${RESPONSE_DELAY}"]