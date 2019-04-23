import subprocess
import threading
import sys
import queue
import os 
import time


ROOT_DIR = os.path.abspath('./')
sys.path.append(ROOT_DIR)


def read_output(pipe, q):

    while True:
        l = pipe.readline()
        q.put(l)

proc_a = subprocess.Popen(["stdbuf", "-o0", "python", 'main.py', 'video', '--device=0', '--vid_path='+ ROOT_DIR + 'dataset/videos/YoutubeVid2.mp4'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
proc_b = subprocess.Popen(["stdbuf", "-o0", './compute_flow', '--gpuID=1', '--type=1', '--skip=1', '--vid_path=' + ROOT_DIR + 'dataset/videos/YoutubeVid2.mp4', '--out_path=/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/dataset/output'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE)

proc_c = subprocess.Popen(["stdbuf", "-o0", 'python', 'main.py', 'predict', '--device=1', '--spatial=/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/weights/spatial/009-0.298.hdf5', '--temporal=/home/dmitry/Documents/Projects/opticalFlow_TwoStreamNN/weights/temporal/575-0.002.hdf5'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE)



pa_q = queue.Queue()
pb_q = queue.Queue()
pc_q = queue.Queue()

pb_t = threading.Thread(target=read_output, args=(proc_b.stdout, pb_q))
pa_t = threading.Thread(target=read_output, args=(proc_a.stdout, pa_q))
pc_t = threading.Thread(target=read_output, args=(proc_c.stdout, pc_q))

pa_t.daemon = True
pb_t.daemon = True
pc_t.daemon = True 

pb_t.start()
pa_t.start()
time.sleep(10)
pc_t.start()

while True:

    proc_a.poll()
    proc_b.poll()
    proc_c.poll()

    if proc_a.returncode is not None or proc_b.returncode is not None:
        break

    # write output from procedure A (if there is any)
    # try:
    #     l = pa_q.get(False)
    #     sys.stdout.write("A: ")
    #     sys.stdout.write(l)
    # except queue.Empty:
    #     pass

    # # write output from procedure B (if there is any)
    # try:
    #     l = pb_q.get(False)
    #     sys.stdout.write("B: ")
    #     sys.stdout.write(l)
    # except queue.Empty:
    #     pass