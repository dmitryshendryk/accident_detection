import json 
import os 
import sys
import requests
from tools.config import Config

import datetime

from bson import json_util
import json

ROOT_DIR = os.path.abspath('../')
sys.path.append(ROOT_DIR)



class RestAPI():

    def __init__(self):
        self.config = Config()

    def send_post(self):
        headers = {}
        payload = {"cameraId": "1476320433439","type": "Danger","message": "Accident detected","cameraViewImageUrl": "link_to_detected_image", "eventTime": str(datetime.datetime.now())}
        r = requests.post(self.config.POST_URL, json=(payload), headers=headers)
        print("Response {}".format(r))

