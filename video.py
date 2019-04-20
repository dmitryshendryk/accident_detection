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


from scipy import misc 

ROOT_DIR = os.path.abspath('./')
sys.path.append(ROOT_DIR)

import flowfilter.plot as fplot
import flowfilter.gpu.flowfilters as gpufilter

from subprocess import Popen, PIPE 
import subprocess
from threading import Thread
from queue import Queue, Empty
from tools.server_zmq import ServerTask


def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

def demo(yolo, vid_path):

    # server = ServerTask()
    # server.start()

    # ON_POSIX = 'posix' in sys.builtin_module_names
   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # gpuF = gpufilter.PyramidalFlowFilter(480, 640, 2)
    # gpuF.gamma = [10, 50]                                   # gains for each level
    # gpuF.maxflow = 4.0                                      # maximum optical flow value
    # gpuF.smoothIterations = [2, 4]                          # smooth iterations per level
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = False 
    
    video_capture = cv2.VideoCapture(os.path.join(ROOT_DIR, 'dataset/videos/YoutubeVid2.mp4'))
    # p = subprocess.check_output("./compute_flow --gpuID=1 --type=1 --skip=100 --vid_path=/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/dataset/videos/YoutubeVid2.mp4 --out_path=/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/dataset/output",
                    # shell=True)
    # p = Popen(['./compute_flow --gpuID=0 --type=1 --skip=10 --vid_path= /home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/dataset/videos/YoutubeVid2.mp4 --out_path=/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/dataset/output'],
                # shell=True, stdout=PIPE)
    # q = Queue()
    # t = Thread(target=enqueue_output, args=(p.stdout, q))
    # t.daemon = True
    # t.start()


    fps = 0.0
    i_frame = 0
    i_folder = 0
    os.makedirs(os.path.join(ROOT_DIR, 'dataset/output_rgb/' + "%06d"%i_folder))
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()
        frame_flow = frame.copy()

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
        
        # cv2.namedWindow("Optical flow", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        # Compute optical flow 
        # img = cv2.cvtColor(frame_flow, cv2.COLOR_BGR2GRAY)
        # img = misc.imresize(img, (480, 640))
        # gpuF.loadImage(img)
        # gpuF.compute()
        # flow = gpuF.getFlow()

        # rgb_frame_out.write(frame)
        # opt_flow = fplot.flowToColor(flow, 3.0)
        
        # optical_frame_out.write(opt_flow)
        # cv2.imwrite("frame%d.jpg" % count, frame)
        # cv2.imwrite("optical_frame%d.jpg" % count, opt_flow)

        # try:
        #     line = q.get_nowait()
            # std_buf = StringIO(line)
            # print(std_buf)
            # line = line.decode().replace('\n','').replace(';','').replace(' ','').split(',')
            # if len(line) > 3:
            #     line = list(map(int, line[2:]))
            #     line = np.array(line)
            #     print("Result {}".format(line.reshape(640,1240)))

        # except Empty:
            # pass
            # print('no output yet')
        # else:
            # pass

        # sub_results = p.decode("utf-8")
        # print("Result: {}".format(sub_results))
        # cv2.imshow("Optical flow", fplot.flowToColor(flow, 3.0))

        cv2.imshow('Tracking', frame)
        
        if i_frame == 20:
            i_folder += 1
            if not os.path.exists(os.path.join(ROOT_DIR, 'dataset/output/rgb/' + "%06d"%i_folder)):
                os.makedirs(os.path.join(ROOT_DIR, 'dataset/output/rgb/' + "%06d"%i_folder))
            i_frame=0
            continue
           
        cv2.imwrite(os.path.join(ROOT_DIR, 'dataset/output/rgb/' + "%06d"%i_folder + '/frame' + "%06d"%i_frame + '.jpg') , frame)
        i_frame += 1
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        # print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()
