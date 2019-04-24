import argparse

from video import demo 
from yolo import YOLO
from two_stream_network.spatial_train import train_spatial
from two_stream_network.temporal_train import train_temporal 
from two_stream_network.spatial_validate import spatial_validate
from two_stream_network.fuse_validate import fuse_train
from two_stream_network.temporal_validate import validate_temporal
from two_stream_network.fuse_predict import fuse_prediction
from tools.utils import create_csv
from tensorflow.python.client import device_lib

import subprocess
import threading
import sys
import queue
import os 
import time
import pathlib
import shutil

import os 
ROOT_DIR = os.path.abspath('./')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def read_output(pipe, q):

    while True:
        l = pipe.readline()
        q.put(l)

def create_folders():
    if os.path.exists(ROOT_DIR + '/dataset/output'):
        shutil.rmtree(ROOT_DIR + '/dataset/output')
    pathlib.Path(ROOT_DIR + '/dataset/output/rgb').mkdir(parents=True, exist_ok=True)
    pathlib.Path(ROOT_DIR + '/dataset/output/v').mkdir(parents=True, exist_ok=True)
    pathlib.Path(ROOT_DIR + '/dataset/output/u').mkdir(parents=True, exist_ok=True)


def start_detection(vid_path, spatial_weights, temporal_weights):

    proc_a = subprocess.Popen(["stdbuf", "-o0", "python", 'main.py', 'video', '--device=0', '--vid_path='+ vid_path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc_b = subprocess.Popen(["stdbuf", "-o0", './compute_flow', '--gpuID=1', '--type=1', '--skip=1', '--vid_path=' + vid_path, '--out_path= ' + ROOT_DIR + '/dataset/output'],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

   

    pa_q = queue.Queue()
    pb_q = queue.Queue()

    pb_t = threading.Thread(target=read_output, args=(proc_b.stdout, pb_q))
    pa_t = threading.Thread(target=read_output, args=(proc_a.stdout, pa_q))

    pa_t.daemon = True
    pb_t.daemon = True

    pb_t.start()
    pa_t.start()

    time.sleep(20)

    proc_c = subprocess.Popen(["stdbuf", "-o0", 'python', 'main.py', 'predict', '--device=1', '--spatial=' + spatial_weights, '--temporal=' + temporal_weights],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    pc_q = queue.Queue()
   
    pc_t = threading.Thread(target=read_output, args=(proc_c.stdout, pc_q))

    pc_t.daemon = True 

    pc_t.start()

    while True:
        try:
            proc_b.poll()
            proc_a.poll()
            proc_c.poll()

            if proc_a.returncode is not None or proc_b.returncode is not None:
                break
        except KeyboardInterrupt:
            try:
                proc_a.terminate()
                proc_b.terminate()
                proc_c.terminate()
            except OSError:
                pass
        
        try:
            l = pc_q.get(False)
            sys.stdout.write("B: ")
            sys.stdout.write(l.decode())
        except queue.Empty:
            pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="main fucntion"
    )

    parser.add_argument('command', metavar='<command>', help="''")
    parser.add_argument('--spatial')
    parser.add_argument('--temporal')
    parser.add_argument('--vid_path')
    parser.add_argument('--device')
    parser.add_argument('--train_path')


    args = parser.parse_args()

    if args.command == 'video':
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
    
    if args.command == 'create_csv':
        create_csv(args.train_path)

    if args.command == 'predict':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        fuse_prediction(args.spatial, args.temporal)
    
    if args.command == 'predict_local':
        create_folders()
        start_detection(args.vid_path, args.spatial, args.temporal)
