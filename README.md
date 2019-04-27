
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