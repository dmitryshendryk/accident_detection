import os 
import sys 
import argparse
import subprocess as sub 


ROOT_DIR = os.path.abspath('./')



if __name__ = "__main__":

    parser = =argparse.ArgumentParser()

    parser.add_argument('--start_frame')
    parser.add_argument('--folder')

    args = parser.parse_args()

    os.makedirs(os.path.join(ROOT_DIR + '/opt_flow_accidents', args.folder))
    sub.call(['ffmpeg -r 20 -f image2 -s 320x240  -start_number  ' + args.start_frame + '  -i  /Users/dmitry/Documents/Business/Projects/Upwork/TectumAI/project/dataset_test/101-200/' + folder + '/%d.jpg -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4'])




    
