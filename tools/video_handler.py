from threading import Thread
import cv2
import time

class VideoStream:
	def __init__(self, src=0, name="VideoStream"):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.src = src
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the thread name
		self.name = name

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, name=self.name, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):

		# keep looping infinitely until the thread is stopped
		while True:

			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
			# print("LOG: updated camera: ", self.name )

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):

		if(self.stopped):
			return
		# indicate that the thread should be stopped
		self.stopped = True
		time.sleep(2)
		self.stream.release()


# if __name__ == "__main__":
#     stream = VideoStream("rtsp://admin:12345abc@92.14.11.106:554/Streaming/Channels/1")
#     while 1:
#         image = stream.read()
#         cv2.imshow('frame', image)
#         cv2.waitKey(0)