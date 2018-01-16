"""
Used to convert and create labeled datasets pickles for training
We use the NIST SD19 by_class dataset
"""
import os
import time
import pickle
import collections
import argparse
from PIL import Image
import numpy as np

HEIGHT = 32
WIDTH = 32
LABELFILES = collections.namedtuple('Label_Files', ['Directory',
                                                    'Training',
                                                    'Testing'])


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

    im = Image.open(image_path)
    pixels = list(im.getdata())

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

            # Write to the testing file
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

    # Return the root dataset directory, training file name, and
    # testing file name
    result = LABELFILES(directory, training_file_name, test_file_name)
    return result


def pickle_data(directory, text_file, datafile):
    """
    Read a text file and pickle the contents to prevent work repeat
    Input: Text file containing the filename, absolute image path, label
    Output: Pickled file of the character data
    """
    # Point is a set that contains the image file name, its path,
    # the label, and the actual image pixel array.
    point = {}

    # Char_data is the list of all points from the label text file
    char_data = []

    start = time.time()

    # First, change to the dataset root directory
    os.chdir(directory)

    try:
        # Open the text file
        with open(text_file) as train:
            for line in train:
                # Remove the new line character from each line
                v = line.strip().split(',')

                # Filename is the first element of the set
                point['filename'] = v[0]

                # Image path is the second element
                point['image_path'] = v[1]

                # Image label is the thrid element
                point['label'] = v[2]

                # Converted image pixel array is the last element of the set
                point['pixel_array'] = convert_to_pixel_array(v[1])

                # Append the set to the list
                char_data.append(point)

        # The complete list is then pickled into the specified filename
        pickle.dump(char_data, open(datafile, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)
        print("> Pickle time: {}".format(time.time() - start))

        # Return the pickle file name
        return datafile

    # Throw an error if we can't find the text file
    except FileNotFoundError:
        print('Label file not found at path {}'.format(directory))


def load_data(directory, datafile):
    """
    Function to load a pickled datafile and return its contents
    Input: Directory to the pickle file, Pickle filename
    Output: Contents of specified pickle file
    """
    # Change to the specified directory
    os.chdir(directory)

    try:
        # Load the pickled file
        data = pickle.load(open(datafile, "rb"))

        # Return the contents
        print('Data loaded from {}'.format(datafile))
        return data

    # Throw an error if we can't find the pickle file
    except FileNotFoundError:
        print('Pickle file {} not found at path {}'
              .format(datafile, directory))


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
    resize_path = resize_images(ARGS['directory'])

    # # Second, generate the label text files
    labels_path = generate_label_list(resize_path)

    # # Pickle the data
    train_pickle = pickle_data(labels_path.Directory,
                               labels_path.Training,
                               "trainingg_set.p")

    test_pickle = pickle_data(labels_path.Directory,
                              labels_path.Testing,
                              "testing_set.p")

    # Load the pickle data
    train = load_data(labels_path.Directory, train_pickle)
    test = load_data(labels_path.Directory, test_pickle)

    for entry in train:
        print(entry)

    for entry in test:
        print(entry)
