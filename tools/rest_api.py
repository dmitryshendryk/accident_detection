import json 
import os 
import requests
from config import Config


class RestAPI():

    def __init__(self):
        self.config = Config()



    def send_post(self, data):
        headers = {'Authorization': 'Bearer ' + self.config.TOKEN}
        payload = data 

        r = requests.post(self.config.POST_URL, json=(payload), headers=headers)