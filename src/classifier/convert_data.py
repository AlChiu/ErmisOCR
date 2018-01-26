"""
Used to convert and create labeled datasets pickles for training
We use the NIST SD19 by_class dataset
"""
import os
import time
import pickle
import argparse
from PIL import Image
import numpy as np
import nist_by_class

HEIGHT = 32
WIDTH = 32
LABELS = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 9,
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
                image = Image.open(image_path).convert('L')
                width, height = image.size

                # Grab the filename
                fname = os.path.splitext(image_path)[0] + '_' + label + '.png'

                if width != WIDTH | height != HEIGHT:
                    # Resize the image from 128 x 128 to 32 x 32
                    resized_image = image.resize((WIDTH, HEIGHT),
                                                 Image.LANCZOS)

                    # Save the new image and replace the old image
                    resized_image.save(fname, 'PNG')

                    # Print the progress
                    print('Resized ' + item)

                    # Remove the old image
                    os.remove(image_path)

                else:
                    print(item + ' is already the correct size')

    # Time
    print('> Completion Time: {}'.format(time.time() - start))
    return directory


def convert_to_pixel_array(image_path):
    """
    Convert an image into a 2D array of pixel intensities, standardized
    and zero-centered by subtracting each pixel value by the image mean
    and dividing it by the stardard deviation.
    Input: Absolute image path
    Output: Resized, normalized pixel array of the image
    """
    pixels = []

    iamge = Image.open(image_path)
    pixels = list(iamge.getdata())

    # Normalize and zero center pixel data
    std_dev = np.std(pixels)
    mean = np.mean(pixels)

    # Each pixel will have the image mean substracted from it.
    # Then divide the result by the image standard deviation.
    # Replace that result as the new pixel value
    pixels = [(pixels[offset:offset+WIDTH] - mean)/std_dev for offset in
              range(0, WIDTH*HEIGHT, WIDTH)]

    # Convert the pixel list into a numpy array
    pixels = np.array(pixels).astype(np.float32)

    return pixels


def generate_label_list(directory):
    """
    Generate a text file that lists all of the image paths with their label
    Input: NIST_split directory (after running nist_by_class script)
    Output: text file of image paths with their label
    """
    start = time.time()

    # Change to the directory
    os.chdir(directory)

    # List out subset folders
    root_dirc = os.listdir()

    for subset in root_dirc:
        if os.path.isdir(subset):
            label_path = os.path.join(directory, subset)
            labels = os.listdir(label_path)

            train_file = "char_train_label_" + str(subset) + ".txt"
            test_file = "char_test_label_" + str(subset) + ".txt"

            train_file = open(os.path.join(directory, train_file), "w+")
            test_file = open(os.path.join(directory, test_file), "w+")

            for label in labels:
                folder_path = os.path.join(label_path, label)
                folders = os.listdir(folder_path)

                for folder in folders:
                    image_path = os.path.join(folder_path, folder)
                    images = os.listdir(image_path)

                    for image in images:
                        img_path = os.path.join(image_path, image)
                        if folder == 'training':
                            train_file.write(str(image) +
                                             "," +
                                             str(img_path) +
                                             "," +
                                             str(label) +
                                             "\r\n")
                        if folder == 'testing':
                            test_file.write(str(image) +
                                            "," +
                                            str(img_path) +
                                            "," +
                                            str(label) +
                                            "\r\n")

    train_file.close()
    test_file.close()

    # Time
    print('> Label files complete: {}'.format(time.time() - start))
    return directory


def pickle_data(directory, text_file_no, datafile, setting):
    """
    Read a text file and pickle the contents to prevent work repeat
    Input: Text file containing the filename, absolute image path, label
    Output: Pickled file of the character data
    """
    start = time.time()
    # Point is a dictionary that contains the image file name, its path,
    # the label, and the actual image pixel array.

    # Char_data is the dictionary of all points from the label text file
    char_data = []

    # First, change to the dataset root directory
    os.chdir(directory)

    # Find the text file
    if setting == "training":
        text_file = "char_train_label_subset_" + str(text_file_no) + ".txt"
    elif setting == "testing":
        text_file = "char_test_label_subset_" + str(text_file_no) + ".txt"
    else:
        print("We need the correct setting")

    # Open the text file
    if text_file:
        try:
            with open(text_file) as text:
                for line in text:
                    # Remove the new line character from each line
                    text_line = line.strip().split(',')
                    image = convert_to_pixel_array(text_line[1])
                    text_line[2] = LABELS.get(text_line[2])
                    text_line.append(image)
                    char_data.append(text_line)
            pickle.dump(char_data, open(datafile, "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
        except FileNotFoundError:
            print('Text file {} not foud'.format(text_file))

    print("> Pickle time: {}".format(time.time() - start))


def load_data(directory, datafile_no, setting):
    """
    Function to load a pickled datafile and return its contents
    Input: Directory to the pickle file, subset no, setting(testing/training)
    Output: Contents of specified pickle file
    """
    os.chdir(directory)

    # Based on setting, load the specified pkl file
    if setting == "testing":
        fname = "test_pickle_" + str(datafile_no) + ".pkl"
    elif setting == "training":
        fname = "train_pickle_" + str(datafile_no) + ".pkl"
    else:
        print("The setting must be specified as testing or training")

    # Unpickle the pickle
    if fname:
        try:
            data = pickle.load(open(fname, "rb"))
            print('Data loaded from {}'.format(fname))
        except FileNotFoundError:
            print('{} not found'.format(fname))
            print('Creating {}'.format(fname))
            pickle_data(directory, datafile_no, fname, setting)
            data = pickle.load(open(fname, "rb"))
            print('Data loaded from {}'.format(fname))

    return data


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", help="path to dataset")
    AP.add_argument("-t", "--text", help="path to text_file")
    ARGS = vars(AP.parse_args())

    #############################
    # Main pipeline test
    #############################

    # First, resize the images
    RESIZED = resize_images(ARGS['directory'])

    # Second, spilt the dataset into multiple subsets
    SPLIT, DIVISION = nist_by_class.split_NIST(RESIZED, 10)

    # Third, generate the label text files
    DATA = generate_label_list(SPLIT)

    # Pickle the ten subsets
    COUNT = 0
    while COUNT < DIVISION:
        TRAINNAME = "train_pickle_" + str(COUNT) + ".pkl"
        TESTNAME = "test_pickle_" + str(COUNT) + ".pkl"
        pickle_data(DATA, COUNT, TRAINNAME, "training")
        pickle_data(DATA, COUNT, TESTNAME, "testing")
        COUNT += 1

    # # Load the pickle data
    TRAIN = load_data(DATA, 9, "training")
    TEST = load_data(DATA, 9, "testing")
