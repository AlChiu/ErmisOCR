"""
Used to convert and create labeled datasets pickles for training
We use the NIST SD19 by_class dataset
"""
import os
import time
import math
import argparse
import cv2
from scipy import ndimage
import numpy as np
import nist_by_class

HEIGHT = 224
WIDTH = 224


def get_best_shift(img):
    """
    Using scipy, we can calculate the center of mass of
    the input image and return the best shift amount to
    center an image on the character.
    """
    center_y, center_x = ndimage.measurements.center_of_mass(img)

    rows, columns = img.shape
    shift_x = np.round(columns/2.0 - center_x).astype(int)
    shift_y = np.round(rows/2.0 - center_y).astype(int)

    return shift_x, shift_y


def shift(img, sft_x, sft_y):
    """
    Shift the input image by sft_x and sft_y pixels
    """
    rows, columns = img.shape
    trans = np.float32([[1, 0, sft_x], [0, 1, sft_y]])
    shifted_image = cv2.warpAffine(img, trans, (columns, rows))
    return shifted_image


def resize_shift(image):
    """
    Resize an image into a 224 x 224 image using LeCun's
    preprocessing technique.
    """
    (_, image) = cv2.threshold(image, 127, 255,
                               cv2.THRESH_BINARY_INV |
                               cv2.THRESH_OTSU)

    # Remove rows and columns that are all black
    while np.sum(image[0]) == 0:
        image = image[1:]

    while np.sum(image[:, 0]) == 0:
        image = np.delete(image, 0, 1)

    while np.sum(image[-1]) == 0:
        image = image[:-1]

    while np.sum(image[: -1]) == 0:
        image = np.delete(image, -1, 1)

    rows, cols = image.shape

    # Resize the resulting image into a 224 x 224 image
    # while maintaining the aspect ratio
    if rows > cols:
        factor = 214.0 / rows
        rows = 214
        cols = int(round(cols * factor))
    else:
        factor = 214.0 / cols
        cols = 214
        rows = int(round(rows * factor))
    image = cv2.resize(image, (cols, rows))

    # Pad the 214 x 214 so that it is now 224 x 224
    column_padding = (int(math.ceil((224-cols)/2.0)),
                      int(math.ceil((224-cols)/2.0)))
    row_padding = (int(math.ceil((224-rows)/2.0)),
                   int(math.ceil((224-rows)/2.0)))
    image = np.lib.pad(image,
                       (row_padding, column_padding),
                       'constant')

    # Center the character in this new image
    x_shift, y_shift = get_best_shift(image)
    shifted = shift(image, x_shift, y_shift)
    image = shifted
    return image


def resize_images(directory):
    """
    Resize NIST images to 224 x 224
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
                image = cv2.imread(image_path, 0)
                width, height = image.shape

                # Grab the filename
                fname = os.path.splitext(image_path)[0] + '_' + label + '.png'

                if width != WIDTH or height != HEIGHT:
                    # Resize the image from 128 x 128 to 32 x 32 using
                    # Lecun's method seen in the MNIST dataset

                    image = resize_shift(image)

                    # Save the image
                    cv2.imwrite(fname, image)
                    os.remove(image_path)
                    print('{} resized'.format(item))

                else:
                    print(item + ' is already the correct size')

    # Time
    print('> Completion Time: {}'.format(time.time() - start))
    return directory


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", help="path to dataset")
    AP.add_argument("-t", "--text", help="path to text_file")
    ARGS = vars(AP.parse_args())

    #############################
    # Main pipeline test
    #############################

    # # First, resize the images
    RESIZED = resize_images(ARGS['directory'])

    # # Second, spilt the dataset into multiple subsets
    SPLIT, DIVISION = nist_by_class.split_NIST(RESIZED, 10)
