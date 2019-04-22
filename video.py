#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')
from io import StringIO
import time


from scipy import misc 

ROOT_DIR = os.path.abspath('./')
sys.path.append(ROOT_DIR)

# import flowfilter.plot as fplot
# import flowfilter.gpu.flowfilters as gpufilter

from subprocess import Popen, PIPE 
import subprocess
from threading import Thread
from queue import Queue, Empty
from tools.video_handler import VideoStream

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

def demo(yolo, vid_path):

    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = False 
    
    # video_capture = cv2.VideoCapture(os.path.join(ROOT_DIR, 'dataset/videos/YoutubeVid2.mp4'))
    # video_capture = VideoStream(os.path.join(ROOT_DIR, 'dataset/videos/YoutubeVid2.mp4'))

    video_capture = VideoStream(vid_path)
    fps = 0.0
    i_frame = 0
    i_folder = 0
    os.makedirs(os.path.join(ROOT_DIR, 'dataset/output/rgb/' + "%06d"%i_folder))
    while True:
        frame = video_capture.read()  # frame shape 640*480*3
        # if ret != True:
        #     break
        t1 = time.time()
        # frame_flow = frame.copy()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        # cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)


        # cv2.imshow('Tracking', frame)
        
        if i_frame == 20:
            i_folder += 1
            if not os.path.exists(os.path.join(ROOT_DIR, 'dataset/output/rgb/' + "%06d"%i_folder)):
                os.makedirs(os.path.join(ROOT_DIR, 'dataset/output/rgb/' + "%06d"%i_folder))
            i_frame=0
            continue
           
        cv2.imwrite(os.path.join(ROOT_DIR, 'dataset/output/rgb/' + "%06d"%i_folder + '/' + "%05d"%i_frame + '.jpg') , frame)
        i_frame += 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()
