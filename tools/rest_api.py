import json 
import os 
import sys
import requests
from tools.config import Config
# from config import Config
import datetime

import json

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)



class RestAPI():

    def __init__(self):
        self.config = Config()

    def send_post(self, camera_id, image_path):
        headers = {}
        payload = {"cameraId": str(camera_id), "type": "Danger","message": "Accident detected","cameraViewImageUrl": str(image_path), "eventTime": str(datetime.datetime.now())}
        print(payload)
        r = requests.post(self.config.POST_URL, json=(payload), headers=headers)
        print("Response {}".format(r))

if __name__ == "__main__":
    rest = RestAPI()
    rest.send_post()