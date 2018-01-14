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

HEIGHT = 32
WIDTH = 32


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
                fname = os.path.splitext(image_path)[0] + '_' + label + '.png'

                # Resize the image from 128 x 128 to 32 x 32
                imResize = im.resize((WIDTH, HEIGHT), Image.LANCZOS)

                # Save the new image and replace the old image
                imResize.save(fname, 'PNG')

                # Print the progress
                print('Resized ' + item)

                # Remove the old image
                os.remove(image_path)

    # Time
    print('> Completion Time: {}'.format(time.time() - start))


def convert_to_pixel_array(image_path):
    """
    Convert an image into a 2D array of pixel intensities, standardized
    and zero-centered by subtracting each pixel value by the image mean
    and dividing it by the stardard deviation.
    Input: Absolute image path
    Output: Resized, normalized pixel array of the image
    """
    pixels = []

    im = Image.open(image_path)
    pixels = list(im.getdata())

    # Normalize and zero center pixel data
    std_dev = np.std(pixels)
    mean = np.mean(pixels)

    pixels = [(pixels[offset:offset+WIDTH] - mean)/std_dev for offset in
              range(0, WIDTH*HEIGHT, WIDTH)]
    pixels = np.array(pixels).astype(np.float32)

    return pixels


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
                    train_file.write(str(image) +
                                     "," +
                                     str(image_path) +
                                     ","
                                     + str(label)
                                     + "\r\n")

            # Wrote tp the testing file
            if folder == 'testing':
                folder_path = os.path.join(label_path, folder)
                os.chdir(folder_path)
                folder_dirc = os.listdir()
                for image in folder_dirc:
                    image_path = os.path.join(folder_path, image)
                    test_file.write(str(image) +
                                    "," +
                                    str(image_path) +
                                    ","
                                    + str(label)
                                    + "\r\n")
    # Close the files
    train_file.close()
    test_file.close()

    # Time
    print('> Label files complete: {}'.format(time.time() - start))


def pickle_data(text_file, datafile):
    """
    Read a text file and pickle the contents to prevent work repeat
    Input: Text file containing the filename, absolute image path, label
    Output: Pickled file of the character data
    """
    point = {}
    char_data = []
    start = time.time()
    with open(text_file) as train:
        for line in train:
            v = line.strip().split(',')
            point['filename'] = v[0]
            point['image_path'] = v[1]
            point['label'] = v[2]
            point['pixel_array'] = convert_to_pixel_array(v[1])
            char_data.append(point)
    pickle.dump(char_data, open(datafile, "wb"))
    print("> Pickle time: {}".format(time.time() - start))


def load_data(directory):
    for root, dirnames, filenames in os.walk(directory):
        print(root)
        print(dirnames)
        print(filenames)


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", help="path to dataset")
    AP.add_argument("-t", "--text", help="path to text_file")
    ARGS = vars(AP.parse_args())

    # resize_images(ARGS["directory"])
    # generate_label_list(ARGS["directory"])
    pickle_data(ARGS["text"], 'training_set.p')
