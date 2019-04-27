import json 
import os 
import requests
from config import Config

import datetime

class RestAPI():

    def __init__(self):
        self.config = Config()

    def send_post(self):
        headers = {}
        payload = {"cameraId": "1476320433439","type": "Danger","message": "Accident detected","cameraViewImageUrl": "link_to_detected_image", "eventTime": datetime.datetime.now()}
        r = requests.post(self.config.POST_URL, json=(payload), headers=headers)
        print("Response {}".format(r))

