class Config():

    POST_URL = 'http://ec2-18-217-76-76.us-east-2.compute.amazonaws.com/api/v1/ai/message'
    TOKEN = '4cb3d1e9-f6c3-4d47-9ef2-c65b394ac8d7'

    server = 'ec2-18-217-76-76.us-east-2.compute.amazonaws.com,5434'
    database = 'master'
    username = 'AIUSER'
    password = 'AIPassword!q@'
    connecting_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password
    

    camera_table = 'TectumTraffic.ai.Camera'
    camera_server_table = 'TectumTraffic.ai.ServerCamera'    