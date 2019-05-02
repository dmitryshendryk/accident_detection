
#Yolov3

Extract rgb 

```
python main.py video --vid_path=</path/to/video/source>
```


# Optical flow 

### Compile

```
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 -D WITH_CUBLAS=1 -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.5/modules -D BUILD_EXAMPLES=ON ..
```

### Run

```
./compute_flow --gpuID=0 --type=1 --skip=100 --vid_path='/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/dataset/videos' --out_path='/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/dataset/output'
```

# Two Stream Network



1. Train spatial network 

```
python main.py train_spatial
```

2. Train temporal network 

```
python main.py train_temporal
```

3. Validate spatial network 

```
python main.py validate_spatial --spatial=</path/to/weigths>
```

4. Validate temporal network

```
ptyhon main.py validate_temporal --temporal=</path/to/weights>
```

5. Start prediction

```
cd workspace
python main.py detect --weights='<path to the weights in weights dir>' --dataset='<path to the video>'
```

# Server

python3 main.py detect --device=1 --weights=weights/mask_rcnn_accident_0282_v1.h5 --vid_path=videos_accident/cctv_1.mp4

dmitry@35.193.146.105

# Docker 

## Build image
```
sudo nvidia-docker build -t accident_detection  .
```

## Run image

GPU_DEVICE= choose the on which gpu run
STREAM_TYPE= should be camera or video
VID_PATH=path to video if STREAM_TYPE=video
WEIGHTS=weights of the model, default is weights/mask_rcnn_accident_0282_v1.h5


### Run in background
```
sudo nvidia-docker run -e GPU_DEVICE=1 -e STREAM_TYPE=camera -e VID_PATH=videos_accident/cctv_1.mp4 -e WEIGHTS=weights/accident_detection_v_02.h5 -e --response_delay=1 -v ~/accident_detection/imgs:/accident_detection/imgs -d accident_detection
```

### Run with output in front

```
sudo nvidia-docker run -e GPU_DEVICE=1 -e STREAM_TYPE=video -e VID_PATH=videos_accident/cctv_1.mp4 -e WEIGHTS=weights/accident_detection_v_02.h5 -e --response_delay=1 -v ~/accident_detection/imgs:/accident_detection/imgs --rm -ti  accident_detection
```