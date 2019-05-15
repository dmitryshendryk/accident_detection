if [ ! -d model_data ]
then 
    mkdir model_data
fi 

cd model_data

if [ -f coco_classes.txt ]
then 
    echo "File already downloaded, skipping"
else
    wget http://ec2-18-217-76-76.us-east-2.compute.amazonaws.com:8090/file/accident_detection/coco_classes.txt
fi

if [ -f mars-small128.pb ]
then 
    echo "File already downloaded, skipping"
else
    wget http://ec2-18-217-76-76.us-east-2.compute.amazonaws.com:8090/file/accident_detection/mars-small128.pb
fi

if [ -f voc_classes.txt ]
then 
    echo "File already downloaded, skipping"
else
    wget http://ec2-18-217-76-76.us-east-2.compute.amazonaws.com:8090/file/accident_detection/voc_classes.txt
fi

if [ -f yolo_anchors.txt ]
then 
    echo "File already downloaded, skipping"
else
    wget http://ec2-18-217-76-76.us-east-2.compute.amazonaws.com:8090/file/accident_detection/yolo_anchors.txt
fi

if [ -f yolo.h5 ]
then 
    echo "File already downloaded, skipping"
else
    wget http://ec2-18-217-76-76.us-east-2.compute.amazonaws.com:8090/file/accident_detection/yolo.h5
fi

if [ -f video_1_LSTM_1_1024.h5 ]
then 
    echo "File already downloaded, skipping"
else
    wget http://ec2-18-217-76-76.us-east-2.compute.amazonaws.com:8090/file/accident_detection/video_1_LSTM_1_1024.h5
fi

if [ -f vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 ]
then 
    echo "File already downloaded, skipping"
else
    wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
fi


cd ..