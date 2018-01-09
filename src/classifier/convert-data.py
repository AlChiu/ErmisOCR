"""
Used to convert and create labeled datasets pickles for training
We use the NIST SD19 by_class dataset
"""
import os
import time
import shutil
import binascii
import glob
import re
import pickle
import configparser
import argparse
from PIL import Image
import numpy as np


def resize_images(directory):
    """
    Resize NIST images down to 32 x 32
    Input: by_class directory of NIST SD19
    Output: Resized images of NIST SD19 by_class
    """
    start = time.time()
    dirs = os.listdir(directory)
    for label in dirs:
        label_path = os.path.join(directory, label)
        d_list = os.listdir(label_path)
        for folder in d_list:
            data_path = os.path.join(label_path, folder)
            images = os.listdir(data_path)
            for item in images:
                image_path = os.path.join(data_path, item)
                im = Image.open(image_path).convert('L')
                filename = os.path.splitext(image_path)[0]
                imResize = im.resize((32, 32), Image.LANCZOS)
                imResize.save(filename + '.png', 'PNG')
                print('Resized ' + item)
    print('> Completion Time: {}'.format(time.time() - start))

if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", required=True, help="path to dataset")
    ARGS = vars(AP.parse_args())

    resize_images(ARGS["directory"])
