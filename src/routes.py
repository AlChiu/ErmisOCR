"""
Main routes to define the paths the Flask server.
"""
import os
import sys
import argparse
import glob
import cv2
from werkzeug import secure_filename
from classifier import classifier
from detector import char_detect_segment as det_seg

AP = argparse.ArgumentParser()
AP.add_argument("-i", "--image", required=True, help="image")
ARGS = vars(AP.parse_args())

det_seg.detect_segment(ARGS["image"])
