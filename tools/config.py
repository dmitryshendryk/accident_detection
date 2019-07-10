import os
class Config():
    SERVER_URL = "rtmp://" + os.environ['SERVER_URL']
    IMAGE_PORT = os.environ['IMAGE_PORT']
    STREAM_PORT = os.environ['STREAM_PORT']

    POST_URL = 'http://ec2-3-17-203-43.us-east-2.compute.amazonaws.com:8080/api/v1/ai/message'
    TOKEN = '4cb3d1e9-f6c3-4d47-9ef2-c65b394ac8d7'
    IMG_URL_REMOTE = "http://" + os.environ['SERVER_URL'] + ":" + IMAGE_PORT
    IMG_URL_LOCAL = 'http://localhost:8090'

    server = 'ec2-18-223-248-81.us-east-2.compute.amazonaws.com,5434'
    database = 'master'
    username = 'AIUSER'
    password = 'AIPassword!q@'
    connecting_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password


    camera_table = 'TectumTraffic.ai.Camera'
    camera_server_table = 'TectumTraffic.ai.ServerCamera'