import json 
import os 
import requests
from config import Config

import json 

class RestAPI():

    def __init__(self):
        self.config = Config()

    def send_post(self, data):
        headers = {}
        payload = {"cameraId": "1476320433439","type": "Danger","message": "11 No helmet wearing rider","cameraViewImageUrl": "link_to_detected_image", "eventTime": "2019-04-23T18:25:43.511Z"}
        r = requests.post(self.config.POST_URL, json=(payload), headers=headers)


if __name__ =='__main__':
    rest = RestAPI()
    rest.send_post("test")