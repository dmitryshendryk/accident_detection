import argparse

from video import demo 
from yolo import YOLO
from two_stream_network.spatial_train import train_spatial
from two_stream_network.temporal_train import train_temporal 
from two_stream_network.spatial_validate import spatial_validate
from two_stream_network.fuse_validate import fuse_train
from two_stream_network.temporal_validate import validate_temporal
from two_stream_network.fuse_predict import fuse_prediction

import os 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
"0"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="main fucntion"
    )

    parser.add_argument('command', metavar='<command>', help="''")
    parser.add_argument('--spatial')
    parser.add_argument('--temporal')
    parser.add_argument('--vid_path')
    parser.add_argument('--device')


    args = parser.parse_args()

    if args.command == 'demo':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        demo(YOLO(), args.vid_path)
    

    if args.command == 'train_spatial':
        train_spatial()
    
    if args.command == 'validate_spatial':
        spatial_validate(args.spatial, 2)
    
    if args.command == 'validate_temporal':
        validate_temporal(args.temporal, 2)
    
    if args.command == 'train_temporal':
        train_temporal() 
    
    if args.command == 'train_fuse':
        fuse_train(args.spatial, args.temporal)
    
    if args.command == 'predict':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        fuse_prediction(args.spatial, args.temporal)

