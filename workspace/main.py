import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.models import load_model
# from scipy.misc import imread,imresize
from keras import backend as K

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)  # To find local version of the library
from tools.rest_api import RestAPI
from tools.db_connector import DBReader
from tools.video_handler import VideoStream

from yolo import YOLO

import collections
import imageio
import time 
import pathlib
from timeit import default_timer as timer



# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

time_last_db_check = time.time()
DB_CHECK_INTERVAL = 10.0

def check_db_and_update(db_manager, cam_data):
        global time_last_db_check
        time_since_last_check = time.time() - time_last_db_check

        if(time_since_last_check > DB_CHECK_INTERVAL):
            time_last_db_check = time.time()

            is_changed, add_elements, remove_elements, update_elements =  db_manager.db_changed()

            if(not is_changed and len(update_elements) == 0):
                return False

            print("LOG: DB CHANGED!")
            for new_camera_id in  add_elements:
                print("LOG: adding NEW camera id:{}".format(str(new_camera_id)))
                new_camera_info = db_manager.get_camera_info_by_id(new_camera_id)

                if(len(new_camera_info) == 0 ):
                    print("ERROR: could not find camera info from camera_id")
                    continue

                new_connection_string = db_manager.get_connection_string(new_camera_info)

                cam = { 'stream' :  VideoStream(new_connection_string ,name=new_camera_info['Id']),
                        'info' : new_camera_info
                      }
                cam['stream'].start()

                print("     initing camera: ", cam['stream'].name )
                print("     url: ", new_connection_string)

                cam_data[cam['stream'].name] = cam

            for remove_camera in remove_elements:
                print("removing camera: {} ".format(str(remove_camera)))
                if(str(remove_camera) in cam_data):
                    cam_data[str(remove_camera)]["stream"].stop()
                    del(cam_data[str(remove_camera)])


            for up_camera_id in  update_elements:

                print("updating camera of id:{}".format(str(up_camera_id)))

                new_camera_info = db_manager.get_camera_info_by_id(up_camera_id)
                if(len(new_camera_info) == 0 ):
                    print("ERROR: could not find camera info from camera_id")
                    continue

                new_connection_string = db_manager.get_connection_string(new_camera_info)

                if(str(up_camera_id) in cam_data):
                    cam_data[str(up_camera_id)]["stream"].stop()
                    del(cam_data[str(up_camera_id)]["stream"])

                    cam_data[str(up_camera_id)]["stream"] = VideoStream(new_connection_string ,name=new_camera_info['Id'])
                    cam_data[str(up_camera_id)]['stream'].start()
                else:
                    cam = { 'stream' :  VideoStream(new_connection_string ,name=new_camera_info['Id']),
                        'info' : new_camera_info
                         }
                    cam['stream'].start()

                    print("     initing camera: ", cam['stream'].name )
                    print("     url: ", new_connection_string)

                    cam_data[cam['stream'].name] = cam

            return cam_data

        return False


def cameras_init(cameras_list):

    cam_data = {}
    for camera_info in cameras_list:
        camera_path = Config.SERVER_URL + ':' + Config.STREAM_PORT + '/stream/' + camera_info['Name']
        # camera_path = "rtsp://" + camera_info['CameraUser'] + ':' + camera_info['Password'] + '@' + camera_info['Ip'] + '/Streaming/Channels/1'
        cam = {
            "stream": VideoStream(camera_path, name=camera_info['Id']),
            "info":  camera_info
        }

        cam["stream"].start()

        print("start camera: {}".format(cam["stream"].name))
        cam_data[cam["stream"].name] = cam
    
    return cam_data

def calc_time_elapsed(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Processing time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def detection(db, lstm, yolo, base_model, accident_threshold=70, image_path=None, video_path=None, cam_data=None, response_delay=None):
    # assert image_path or video_path
    class_names = ['BG','accident']
    # Image or video?
    pathlib.Path(ROOT_DIR + '/imgs').mkdir(parents=True, exist_ok=True)
    rest = RestAPI()
    padding_left, padding_right = 50,50
    start = time.time()
    print("VIDEO PATH :  ", video_path)
    print("CAMERA : ", cam_data)
    images_queue = collections.deque(maxlen=40)

    time_last_check = time.time()
    check_interval = 2.0

    if cam_data:
        print("Processing on camera")
        x = []
        while True:
            db_update = check_db_and_update(db, cam_data)
            if db_update:
                cam_data = db_update
            for key in cam_data.keys():
                prev_mag = 0
                prev_varience = 0
                mag = 0
                varience = 0
                camera = cam_data[key]
                image = camera['stream'].read()
                if image is not None:
                    first_frame = image.copy()

                    first_frame = cv2.resize(first_frame,(224,224))
                    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
                    mask = np.zeros_like(first_frame)
                    mask[..., 1] = 255

                    for idx_frame in range(2):

                        print("Process camera {}".format(camera['stream'].name))

                        image = camera['stream'].read()

                        if image is None:
                            print("Frame is broken")
                            # exit(0)
                            continue

                        next_frame = image.copy()

                        next_frame = cv2.resize(next_frame,(224,224))
                        gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
                        
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                        # skip_frame += 1
                        # if skip_frame % 5 == 0:

                        change_mag = abs(mag - prev_mag)
                        binary_mag = np.ones(change_mag.shape,dtype=np.float64)
                        threshold_new = np.mean(change_mag , dtype=np.float64)
                        varience = np.where(change_mag < threshold_new,0,binary_mag)

                        varience = len(varience[varience == 1])

                        if np.all(prev_varience) == 0 and np.all(prev_mag) == 0: 
                            # prev_binary_indicator = binary_indicator
                            prev_varience = varience
                            prev_mag = mag
                            continue

                        if varience / prev_varience >= 3 and varience > prev_varience:
                            print('Potential accident, {}'.format(datetime.datetime.now()))
                        
                            frame_img = Image.fromarray(image[...,::-1])

                            start_time = timer()
                            print('Process yolo')
                            boxs = yolo.detect_image(frame_img)
                            print(boxs)
                            print("Data in x vector {}".format(len(x)))
                            if len(boxs) != 0:
                                for box in boxs:
                                    
                                    frame_img = image[box[1]-padding_left:box[1]+box[3] + padding_left ,box[0] - padding_right:box[0]+box[2] + padding_right]
                                    if frame_img.shape[0] != 0 and frame_img.shape[1] != 0:
                                        frame_img = cv2.resize(frame_img , (224,224))
                                        x.append(frame_img)
                                        # cv2.imwrite(ROOT_DIR+ '/imgs/' + str(int(time.time())) + '.jpg', image)
                                if len(x) > 10:
                                    x = np.array(x)
                                    base_model.predict(x)
                                    print("LSTM processing")
                                    x_features = base_model.predict(x)
                                    x_features = x_features.reshape(x_features.shape[0], x_features.shape[1]*x_features.shape[2], x_features.shape[3])
                                    answer = lstm.predict(x_features)
                                    
                                    answer = [int(np.round(x)) for x in answer]
                                    

                                    accident_amount =  (answer.count(0)/len(answer)) * 100
                                    normal = (answer.count(1)/len(answer))*100 
                                    print("Probabilities ----------------------------------------------")
                                    print("Accident: {} %".format(accident_amount))
                                    print("Normal: {} %".format(normal))
                                    print(' -----------------------------------------------------------')

                                    if int(accident_amount) > int(accident_threshold):
                                        anserImgs = [a for a,b in zip(x, answer) if b != 1]
                                        print( "Images in accidetns: ", len(anserImgs))
                                        print("Post result")
                                        # for indx, img in enumerate(anserImgs):
                                        
                                        img_name = str(int(time.time()))  + '.gif'
                                        img_path = ROOT_DIR+ '/imgs/' + img_name
                                        # cv2.imwrite(img_path, image)
                                        imageio.mimsave(img_path, images_queue)
                                        
                                        # cv2.imwrite(ROOT_DIR+ '/imgs/' + str(int(time.time()))  + '.jpg', image)
                                        rest.send_post(camera['stream'].name, img_path)

                                        rest.save_img(img_name, img_path)
                                        os.remove(img_path)

                                    answer = []

                                    x = []

                        prev_gray = gray
                        prev_mag = mag
                        prev_varience = varience

        end = time.time()
        calc_time_elapsed(start, end)
        cap.release()
        cv2.destroyAllWindows()
    if video_path:
        print("Processing on video")
        cap = cv2.VideoCapture(video_path)

        first_frame = None
        while first_frame is None or len(first_frame) == 0:
            ret, first_frame = cap.read()
        
        first_frame = cv2.resize(first_frame,(224,224))
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(first_frame)
        mask[..., 1] = 255
        
        prev_mag = 0
        prev_varience = 0

        prev_flow_video = 0
        temp_flows = []
        frame_count = 0

        last_post = timer()
        x = []
        while True:
            ref, image = cap.read()
            
            if ref:
                # continue
                if image is None:
                    print("Frame is broken")
                    continue

                images_queue.append(image)
                next_frame = image.copy()
                frame_count += 1
                next_frame = cv2.resize(next_frame,(224,224))
                gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
                
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

                change_mag = abs(mag - prev_mag)
                binary_mag = np.ones(change_mag.shape,dtype=np.float64)
                threshold_new = np.mean(change_mag , dtype=np.float64)
                varience = np.where(change_mag < threshold_new,0,binary_mag)
                temp_flows.append(varience)


                if frame_count % 4 == 0:
                    flow_video = np.zeros(change_mag.shape,dtype=np.float64)
                    for each_flow in temp_flows:
                        flow_video = flow_video + each_flow
                    flow_video = flow_video / 4

                    flow_video = len(flow_video[flow_video == 1])

                    # print("Current binary: {}  Prev binary {}".format(flow_video, prev_flow_video))
                    if np.all(prev_flow_video) == 0:
                        prev_flow_video = flow_video
                        continue

                    if flow_video / prev_flow_video >= 2 and flow_video / prev_flow_video <= 4 and (flow_video != 0 or prev_flow_video != 0) :

                        time_since_last_check = time.time() - time_last_check
            
                        if time_since_last_check > check_interval:
                            time_last_check = time.time()
                            print('Potential accident, {}'.format(datetime.datetime.now()))

                            frame_img = Image.fromarray(image[...,::-1])
                            start_time = timer()
                            print('Process yolo')
                            boxs = yolo.detect_image(frame_img)
                            print(boxs)
                            print("Data in x vector {}".format(len(x)))
                            if len(boxs) != 0:
                                for box in boxs:
                                    
                                    frame_img = image[box[1]-padding_left:box[1]+box[3] + padding_left ,box[0] - padding_right:box[0]+box[2] + padding_right]
                                    if frame_img.shape[0] != 0 and frame_img.shape[1] != 0:
                                        frame_img = cv2.resize(frame_img , (224,224))
                                        x.append(frame_img)
                                        # cv2.imwrite(ROOT_DIR+ '/imgs/' + str(int(time.time())) + '.jpg', frame_img)
                                if len(x) > 1:
                                    x = np.array(x)
                                    base_model.predict(x)
                                    print("LSTM processing")
                                    x_features = base_model.predict(x)
                                    x_features = x_features.reshape(x_features.shape[0], x_features.shape[1]*x_features.shape[2], x_features.shape[3])
                                    answer = lstm.predict(x_features)
                                    
                                    answer = [int(np.round(x)) for x in answer]
                                    

                                    accident_amount =  (answer.count(0)/len(answer)) * 100
                                    normal = (answer.count(1)/len(answer))*100 
                                    print("Probabilities ----------------------------------------------")
                                    print("Accident: {} %".format(accident_amount))
                                    print("Normal: {} %".format(normal))
                                    print(' -----------------------------------------------------------')

                                    if int(accident_amount) > int(accident_threshold):
                                        anserImgs = [a for a,b in zip(x, answer) if b != 1]
                                        print( "Images in accidetns: ", len(anserImgs))
                                        print("Post result")
                                        # for indx, img in enumerate(anserImgs):
                                        # img_name = str(int(time.time()))  + '.jpg'
                                        img_name = str(int(time.time()))  + '.gif'
                                        img_path = ROOT_DIR+ '/imgs/' + img_name
                                        # cv2.imwrite(img_path, image)
                                        imageio.mimsave(img_path, images_queue)
                                        rest.send_post("1476320433439", img_name)

                                        rest.save_img(img_name, img_path)
                                        os.remove(img_path)

                                    answer = []

                                    x = []
                    
                    prev_flow_video = flow_video
                    temp_flows = []
                    frame_count = 0


                varience = len(varience[varience == 1])

                if np.all(prev_varience) == 0 and np.all(prev_mag) == 0: 
                    prev_varience = varience
                    prev_mag = mag
                    continue

                # if varience / prev_varience >= 2 and varience > prev_varience:
                #     print('Potential accident, {}'.format(datetime.datetime.now()))

                prev_gray = gray
                prev_mag = mag
                prev_varience = varience
 
            else:
                break               

                    
        end = time.time()
        calc_time_elapsed(start, end)
        cap.release()
        cv2.destroyAllWindows()


def load_VGG16_model():
  base_model = VGG16(weights=ROOT_DIR + '/workspace/model_data/' + 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224,224,3))
#   base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
  print ("Model loaded..!")
  print (base_model.summary())
  return base_model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
       description='Accidents ')
    
    parser.add_argument('command',
                    metavar='<command>',
                    help="'train, detect, display_data, display_box, mini_mask, display_anchor'")
   
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory inference dataset')

    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
  

    parser.add_argument('--device')
    parser.add_argument('--vid_path')
    parser.add_argument('--streaming')
    parser.add_argument('--response_delay')
    parser.add_argument('--accident_threshold')
    parser.add_argument('--folder')
    
    args = parser.parse_args()


    if args.command == 'detect':
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        cameras_info = []
        cam_data = None
        db = None
        if args.streaming == 'camera':
            db = DBReader()
            if not db.query_cameras():
                exit(0)
            else:
                cameras_list = db.id_list
                if len(cameras_list) != 0:
                    
                    for camera_id in cameras_list:
                        cameras_info.append(db.get_camera_info_by_id(camera_id))
                    
                    cam_data = cameras_init(cameras_info)
                else:
                    print("No active camers found. Finish job")
                    exit(0)

        vid_path = None 
        if args.streaming == 'video':
            vid_path = os.path.join(ROOT_DIR, args.vid_path)

        

        K.clear_session()

        print("Load base BGG16 model")
        base_model = load_VGG16_model()
        print("Load LSTM model")
        lstm = load_model(os.path.join(ROOT_DIR, args.weights))
        print(lstm.summary())

        yolo = YOLO()
        print("Yolo loaded")
        
        if args.folder == "None":

            detection(db, lstm, yolo, base_model, accident_threshold=args.accident_threshold, image_path=None,
                                    video_path=vid_path, cam_data=cam_data, response_delay=args.response_delay)

        elif args.folder != "None":
            videos = os.listdir(ROOT_DIR + '/' + args.folder)
            for video in videos:
                detection(db, lstm, yolo, base_model, accident_threshold=args.accident_threshold, image_path=None,
                                    video_path=ROOT_DIR+ '/' + video, cam_data=cam_data, response_delay=args.response_delay)
































