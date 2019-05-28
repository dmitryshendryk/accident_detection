# Docker 

## Build image
```
sudo docker build -t accident_detection .
```

## Run image

GPU_DEVICE= choose the on which gpu run
STREAM_TYPE= should be camera or video
VID_PATH=path to video if STREAM_TYPE=video
WEIGHTS=weights of the model, default is weights/mask_rcnn_accident_0282_v1.h5

### Change between cameras and video source

STREAM_TYPE=camera
STREAM_TYPE=video

### Run in background
```
sudo nvidia-docker run -e GPU_DEVICE=1 -e STREAM_TYPE=video -e VID_PATH=videos_accident/YoutubeVid1.mp4 -e WEIGHTS=model_data/video_1_LSTM_1_1024.h5 -e ACCIDENT_THRESHOLD=50  -e --response_delay=1 -v ~/accident_detection/imgs:/accident_detection/imgs -v $(pwd)/model_data:/accident_detection/workspace/model_data -d  accident_detection
```


### Run on camera
```
sudo nvidia-docker run -e GPU_DEVICE=0 -e STREAM_TYPE=camera -e VID_PATH=videos_accint/Castro_Street_Cam.mp4 -e WEIGHTS=workspace/model_data/video_1_LSTM_1_1024.h5 -e ACCIDENT_THRESHOLD=50  -e --response_delay=1 -v ~/accident_detection/imgs:/accident_detection/imgs -v $(pwd)/model_data:/accident_detection/workspace/model_data --rm -ti  accident_detection
```


### Run on video with output in front

```
sudo nvidia-docker run -e GPU_DEVICE=0 -e STREAM_TYPE=video -e VID_PATH=videos_accident/Castro_Street_Cam.mp4 -e WEIGHTS=workspace/model_data/video_1_LSTM_1_1024.h5 -e ACCIDENT_THRESHOLD=50  -e --response_delay=1 -v ~/accident_detection/imgs:/accident_detection/imgs -v $(pwd)/model_data:/accident_detection/workspace/model_data --rm -ti  accident_detection
```

