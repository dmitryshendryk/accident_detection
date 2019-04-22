from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

import os 
import sys
import time 


ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)

from two_stream_network.fuse_validate_model import ResearchModels
from two_stream_network.fuse_predict_data import DataSet
import time
import os.path
from os import makedirs
import numpy as np 

def test_1epoch_fuse(
            class_limit=None, 
            n_snip=5,
            opt_flow_len=10,
            saved_model=None,
            saved_spatial_weights=None,
            saved_temporal_weights=None,
            image_shape=(224, 224),
            original_image_shape=(341, 256),
            batch_size=128,
            fuse_method='average'):

    print("class_limit = ", class_limit)

    # Get the data.
    data = DataSet(
            class_limit=class_limit,
            image_shape=image_shape,
            original_image_shape=original_image_shape,
            n_snip=n_snip,
            opt_flow_len=opt_flow_len,
            batch_size=batch_size
            )
    print(data.data_list)
    idx = 0
    two_stream_fuse = ResearchModels(nb_classes=len(data.classes), n_snip=n_snip, opt_flow_len=opt_flow_len, image_shape=image_shape, saved_model=saved_model, saved_temporal_weights=saved_temporal_weights, saved_spatial_weights=saved_spatial_weights)
    print(two_stream_fuse.model.summary())
    
    while 1:
        if not len(data.data_list):
            print('Folder is empty')
            data.data_list = data.get_data_list()
            time.sleep(10)
            continue
        x_batch = data.prediction_iterator(idx)

        predictions = two_stream_fuse.model.predict(x_batch)
        print(predictions)

        data.clean_data_list()
        data.data_list = data.get_data_list()
        time.sleep(10)

def fuse_prediction(saved_spatial_weights,saved_temporal_weights):
    saved_spatial_weights = saved_spatial_weights
    saved_temporal_weights = saved_temporal_weights
    class_limit = None 
    n_snip = 5 # number of chunks used for each video
    opt_flow_len = 10 # number of optical flow frames used
    image_shape=(224, 224)
    original_image_shape=(341, 256)
    batch_size = 2
    fuse_method = 'average'

    test_1epoch_fuse(
            class_limit=class_limit, 
            n_snip=n_snip,
            opt_flow_len=opt_flow_len,
            saved_spatial_weights=saved_spatial_weights,
            saved_temporal_weights=saved_temporal_weights,
            image_shape=image_shape,
            original_image_shape=original_image_shape,
            batch_size=batch_size,
            fuse_method=fuse_method
            )
