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

    # Go through each label folder
    for label in dirs:
        label_path = os.path.join(directory, label)
        d_list = os.listdir(label_path)

        # Go through testing and training folders
        for folder in d_list:
            data_path = os.path.join(label_path, folder)
            images = os.listdir(data_path)

            # Resize each image
            for item in images:
                # Get the absolute path of the image
                image_path = os.path.join(data_path, item)

                # Open the image and convert to grayscale
                im = Image.open(image_path).convert('L')

                # Grab the filename
                filename = os.path.splitext(image_path)[0]

                # Resize the image from 128 x 128 to 32 x 32
                imResize = im.resize((32, 32), Image.LANCZOS)

                # Save the new image and replace the old image
                imResize.save(filename + '.png', 'PNG')

                # Print the progress
                print('Resized ' + item)

    # Time
    print('> Completion Time: {}'.format(time.time() - start))


def generate_label_list(directory):
    """
    Generate a text file that lists all of the image paths with their label
    Input: by_class directory (after running nist_by_class script)
    Output: text file of image paths with their label
    """
    os.chdir(directory)
    root_dirc = os.listdir()

    # Create the text filename
    training_file_name = "char_training_label.txt"
    test_file_name = "char_testing_label.txt"

    # Create the label files
    start = time.time()

    # Open the file, create if they do not exist
    train_file = open(os.path.join(os.pardir, training_file_name), "w+")
    test_file = open(os.path.join(os.pardir, test_file_name), "w+")

    # Go through all of the label folders
    for label in root_dirc:
        label_path = os.path.join(directory, label)
        os.chdir(label_path)
        label_dirc = os.listdir()

        # Each label folder has two folders, training and testing
        for folder in label_dirc:
            # Write to the training file
            if folder == 'training':
                folder_path = os.path.join(label_path, folder)
                os.chdir(folder_path)
                folder_dirc = os.listdir()
                for image in folder_dirc:
                    image_path = os.path.join(folder_path, image)
                    train_file.write(str(image_path)
                                     + ","
                                     + str(label)
                                     + "\r\n")

            # Wrote tp the testing file
            if folder == 'testing':
                folder_path = os.path.join(label_path, folder)
                os.chdir(folder_path)
                folder_dirc = os.listdir()
                for image in folder_dirc:
                    image_path = os.path.join(folder_path, image)
                    test_file.write(str(image_path)
                                    + ","
                                    + str(label)
                                    + "\r\n")
    # Close the files
    train_file.close()
    test_file.close()

    # Time
    print('> Label files complete: {}'.format(time.time() - start))


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", required=True, help="path to dataset")
    ARGS = vars(AP.parse_args())

    #resize_images(ARGS["directory"])
    generate_label_list(ARGS["directory"])
