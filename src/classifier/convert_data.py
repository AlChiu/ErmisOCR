"""
Used to convert and create labeled datasets pickles for training
We use the NIST SD19 by_class dataset
"""
import os
import time
import math
# import pickle
import argparse
import cv2
from scipy import ndimage
import numpy as np
import nist_by_class

HEIGHT = 224
WIDTH = 224
LABELS = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    'a': 10,
    'b': 11,
    'c': 12,
    'd': 13,
    'e': 14,
    'f': 15,
    'g': 16,
    'h': 17,
    'i': 18,
    'j': 19,
    'k': 20,
    'l': 21,
    'm': 22,
    'n': 23,
    'o': 24,
    'p': 25,
    'q': 26,
    'r': 27,
    's': 28,
    't': 29,
    'u': 30,
    'v': 31,
    'w': 32,
    'x': 33,
    'y': 34,
    'z': 35,
    'A': 36,
    'B': 37,
    'C': 38,
    'D': 39,
    'E': 40,
    'F': 41,
    'G': 42,
    'H': 43,
    'I': 44,
    'J': 45,
    'K': 46,
    'L': 47,
    'M': 48,
    'N': 49,
    'O': 50,
    'P': 51,
    'Q': 52,
    'R': 53,
    'S': 54,
    'T': 55,
    'U': 56,
    'V': 57,
    'W': 58,
    'X': 59,
    'Y': 60,
    'Z': 61
}


def getBestShiftAmount(img):
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
    trans_M = np.float32([[1, 0, sft_x], [0, 1, sft_y]])
    shifted_image = cv2.warpAffine(img, trans_M, (columns, rows))
    return shifted_image


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
                image = cv2.imread(image_path, 0)
                width, height = image.shape

                # Grab the filename
                fname = os.path.splitext(image_path)[0] + '_' + label + '.png'

                if width != WIDTH or height != HEIGHT:
                    # Resize the image from 128 x 128 to 32 x 32 using
                    # Lecun's method seen in the MNIST dataset

                    (thresh, image) = cv2.threshold(image, 127, 255,
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

                    # Resize the resulting image into a 28 x 28 image
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

                    # Pad the 28 x 28 so that it is now 32 x 32
                    column_padding = (int(math.ceil((224-cols)/2.0)),
                                      int(math.ceil((224-cols)/2.0)))
                    row_padding = (int(math.ceil((224-rows)/2.0)),
                                   int(math.ceil((224-rows)/2.0)))
                    image = np.lib.pad(image,
                                       (row_padding, column_padding),
                                       'constant')

                    # Center the character in this new image
                    x_shift, y_shift = getBestShiftAmount(image)
                    shifted = shift(image, x_shift, y_shift)
                    image = shifted

                    # Save the image
                    cv2.imwrite(fname, image)
                    os.remove(image_path)
                    print('{} resized'.format(item))

                else:
                    print(item + ' is already the correct size')

    # Time
    print('> Completion Time: {}'.format(time.time() - start))
    return directory


# def generate_label_list(directory):
#     """
#     Generate a text file that lists all of the image paths with their label
#     Input: NIST_split directory (after running nist_by_class script)
#     Output: text file of image paths with their label
#     """
#     start = time.time()

#     # Change to the directory
#     os.chdir(directory)

#     # List out subset folders
#     root_dirc = os.listdir()

#     for subset in root_dirc:
#         if os.path.isdir(subset):
#             label_path = os.path.join(directory, subset)
#             labels = os.listdir(label_path)

#             train_file = "char_train_label_" + str(subset) + ".txt"
#             test_file = "char_test_label_" + str(subset) + ".txt"

#             train_file = open(os.path.join(directory, train_file), "w+")
#             test_file = open(os.path.join(directory, test_file), "w+")

#             for label in labels:
#                 folder_path = os.path.join(label_path, label)
#                 folders = os.listdir(folder_path)

#                 for folder in folders:
#                     image_path = os.path.join(folder_path, folder)
#                     images = os.listdir(image_path)

#                     for image in images:
#                         img_path = os.path.join(image_path, image)
#                         if folder == 'training':
#                             train_file.write(str(image) +
#                                              "," +
#                                              str(img_path) +
#                                              "," +
#                                              str(LABELS.get(str(label))) +
#                                              "\r\n")
#                         if folder == 'testing':
#                             test_file.write(str(image) +
#                                             "," +
#                                             str(img_path) +
#                                             "," +
#                                             str(LABELS.get(str(label))) +
#                                             "\r\n")

#     train_file.close()
#     test_file.close()

#     # Time
#     print('> Label files complete: {}'.format(time.time() - start))
#     return directory


# def pickle_data(directory, text_file_no, datafile, setting):
#     """
#     Read a text file and pickle the contents to prevent work repeat
#     Input: Text file containing the filename, absolute image path, label
#     Output: Pickled file of the character data
#     """
#     start = time.time()
#     # Point is a dictionary that contains the image file name, its path,
#     # the label, and the actual image pixel array.

#     # Char_data is the dictionary of all points from the label text file
#     char_data = []

#     # First, change to the dataset root directory
#     os.chdir(directory)

#     # Find the text file
#     if setting == "training":
#         text_file = "char_train_label_subset_" + str(text_file_no) + ".txt"
#     elif setting == "testing":
#         text_file = "char_test_label_subset_" + str(text_file_no) + ".txt"
#     else:
#         print("We need the correct setting")

#     # Open the text file
#     if text_file:
#         try:
#             with open(text_file) as text:
#                 for line in text:
#                     # Remove the new line character from each line
#                     text_line = line.strip().split(',')
#                     image = cv2.normalize(cv2.imread(text_line[1]),
#                                           None, alpha=0, beta=1,
#                                           norm_type=cv2.NORM_MINMAX,
#                                           dtype=cv2.CV_32F)
#                     char_data.append(LABELS.get(str(text_line[2])))
#                     char_data.append(cv2.resize(image, (224, 224)))
#             pickle.dump(char_data, open(datafile, "wb"),
#                         protocol=pickle.HIGHEST_PROTOCOL)
#         except FileNotFoundError:
#             print('Text file {} not foud'.format(text_file))

#     print("> Pickle time: {}".format(time.time() - start))


# def load_data(directory, datafile_no, setting):
#     """
#     Function to load a pickled datafile and return its contents
#     Input: Directory to the pickle file, subset no, setting(testing/training)
#     Output: Contents of specified pickle file
#     """
#     os.chdir(directory)

#     # Based on setting, load the specified pkl file
#     if setting == "testing":
#         fname = "test_pickle_" + str(datafile_no) + ".pkl"
#     elif setting == "training":
#         fname = "train_pickle_" + str(datafile_no) + ".pkl"
#     else:
#         print("The setting must be specified as testing or training")

#     # Unpickle the pickle
#     if fname:
#         try:
#             data = pickle.load(open(fname, "rb"))
#             print('Data loaded from {}'.format(fname))
#         except FileNotFoundError:
#             print('{} not found'.format(fname))
#             print('Creating {}'.format(fname))
#             pickle_data(directory, datafile_no, fname, setting)
#             data = pickle.load(open(fname, "rb"))
#             print('Data loaded from {}'.format(fname))

#     return data


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

    # # Third, generate the label text files
    # DATA = generate_label_list(SPLIT)
