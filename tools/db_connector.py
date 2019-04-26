import pyodbc
import time
from config import Config

from pypika import Table, Field
from pypika import MSSQLQuery as Query

class DBReader():

    def __init__(self):

        self.init_time = time.time()
        self.config = Config()

        #connect to database
        self.connection = None
        while True:
            try:
                print(self.config.connecting_string)
                self.connection = pyodbc.connect(self.config.connecting_string)
                print(cursor)
                break
            except:
                print("LOG: could not connect to db, retrying...")
                time.sleep(10)



        #dict with camera information from the database
        self.cameras_info = []

        #list with camera idx
        self.id_list =  []

        self.curr_cameras_info = []
        self.curr_id_list = []

        self.query_cameras()

        # init camera information
        self.cameras_info = self.curr_cameras_info
        self.id_list =  self.curr_id_list

    def __del__(self):
        if self.connection is not None:
            self.connection.close()

    def reconnect(self):
        try:
            self.connection = pyodbc.connect(CONNECTION_STRING)
        except:
            print("Lost not connect to db")



    """
    Query database for getting info from cameras

    return: True if succeeded, false if it don't
    """
    def query_cameras(self):

        print("LOG: querying database for camera info !!")
        cursor = self.connection.cursor()

        camera_server_table = Table(CAMERA_SERVER_TABLE)

        #query camera ids with desired service and running on our server
        q = Query.from_(camera_server_table) \
            .select(camera_server_table.cameraId) \
            .where(camera_server_table.serviceId == 3) \
            .where(camera_server_table.serverId == 1) \
            .where(camera_server_table.isActive == '1')

        try:
            cursor.execute(q.get_sql(quote_char=''))
        except:
            print("LOG: FAILED QUERYING DABASET")
            self.curr_cameras_info = []
            self.curr_id_list = []

            return False

        #TODO: can I get this directly as a list
        service_cameras_ids = []
        while True:
            row = cursor.fetchone()
            if not row:
                break
            service_cameras_ids.append(row.cameraId)


        print(service_cameras_ids)

        if(len(service_cameras_ids) == 0):
            self.curr_cameras_info = []
            self.curr_id_list = []
            return True

        print("detected cameras: ", service_cameras_ids)

        camera_table = Table(CAMERA_TABLE)
        q = Query.from_(camera_table).select('*')\
                                     .where(camera_table.Id.isin(service_cameras_ids))\
                                     .where(camera_table.isActive == 1 )
        try:
            cursor.execute(q.get_sql(quote_char=''))
        except:
            print("LOG: FAILED TO QUERY FROM CAMERA TABLE")
            return False

        self.curr_cameras_info = []
        self.curr_id_list = []
        while True:
            row = cursor.fetchone()
            if not row:
                break

            #TODO: this is dependent on how the db is structured
            #      it is not a good commitment to make
            row_dict = { 'Id': row.Id,
                        'Name' : row.Name ,
                        'Latitude':row.Latitude,
                        'Longitude':row.Longitude,
                        'UserId': row.UserId,
                        'CameraUser': row.CameraUser,
                        'Ip': row.Ip,
                        'Password': row.Password,
                        'Address' : row.Address

            }


            self.curr_cameras_info.append(row_dict)
            self.curr_id_list.append(row.Id)

        #close cursor after query
        cursor.close()



        return True

    def get_connection_string(self,camera_info):
        connection_string = "rtsp://" + camera_info['CameraUser']  + ":" + camera_info['Password'] + "@" + camera_info['Ip'] + "/Streaming/Channels/1"
        return connection_string


    def delete_camera_info_by_id(self,id_str):
         # find camera info correspondent to added camera
        new_camera_info = []
        delete_idx = -1
        delete = False
        for idx , camera_info in enumerate(self.cameras_info):
            if(str(camera_info['Id']) == id_str):
                delete = True
                delete_idx = idx
                break

        if(delete):
            del(self.cameras_info[delete_idx])


    def get_camera_info_by_id(self,id_str):
         # find camera info correspondent to added camera
        new_camera_info = []
        for camera_info in self.curr_cameras_info:
            if(str(camera_info['Id']) == id_str):
                new_camera_info = camera_info
                break

        return new_camera_info

    """
    Check if db has changed when new cameras were added or removed

    returns:
        True: bool flag indicating change
        add: list containing new elements
        remove: list containing old elements
    """
    def db_changed(self):

        # #return false if camera query failed
        # if(not self.query_cameras()):
        #      return False , False , False

        self.query_cameras()

        changed = True
        add = []
        remove = []

        # this ignore duplicates
        # TODO: sort can also be used(faster?), or collection.counter
        set_cur = set(self.curr_id_list)
        set_old = set(self.id_list)


        # check if there is a different element in lists
        diff = False
        for d1, d2 in zip(self.cameras_info, self.curr_cameras_info):
            for key, value in d1.items():
                if value != d2[key]:
                    diff = True

        if (set_cur == set_old):
            changed  = False
            return False, add, remove

        union = set(self.curr_id_list) | set(self.id_list)
        add = union - set_old
        remove = union - set_cur

        print("add list: ", list(add))

        self.cameras_info = self.curr_cameras_info
        self.id_list = self.curr_id_list
        return True, list(add), list(remove)

if __name__ == "__main__":
    db = DBReader()