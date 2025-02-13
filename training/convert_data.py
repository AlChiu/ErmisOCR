"""convert_data.py
Used to center and resize images in the NIST SD19 dataset. The dataset
directory needs to be restructured with the nist_restructure.py before
performing the resizing.
"""
import time
import math
import argparse
import pathlib
import cv2
from scipy import ndimage
import numpy as np

HEIGHT = 32
WIDTH = 32


def get_best_shift(img):
    """
    DESCRIPTION: Calculate the center of mass of the input image
    and return the best shift amount to center an image on the character.
    INPUT: Image
    OUTPUT: Calculated shift amount for x and y directions
    """
    # Calculate the center of mass through scipy
    center_y, center_x = ndimage.measurements.center_of_mass(img)

    rows, columns = img.shape
    shift_x = np.round(columns/2.0 - center_x).astype(int)
    shift_y = np.round(rows/2.0 - center_y).astype(int)

    return shift_x, shift_y


def shift(img, sft_x, sft_y):
    """
    DESCRIPTION: Shift the input image by sft_x and sft_y pixels
    INPUT: Image to be shifted, both shift amounts
    OUTPUT: Shifted image
    """
    rows, columns = img.shape

    # Transformation matrix to shift an image by some amount.
    trans = np.float32([[1, 0, sft_x], [0, 1, sft_y]])

    shifted_image = cv2.warpAffine(img, trans, (columns, rows))
    return shifted_image


def preprocess(image):
    """
    DESCRIPTION: Preprocess an image into a 32 x 32 image
    using LeCun's preprocessing technique used for the
    MNIST dataset
    INPUT: Character image
    OUTPUT: Processed character image resized to 32 x 32
    """
    # Invert
    image = cv2.bitwise_not(image)

    # Gaussian Blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Remove rows and columns that sum to zero: ROI Extraction
    while np.sum(image[0]) == 0:
        image = image[1:]

    while np.sum(image[:, 0]) == 0:
        image = np.delete(image, 0, 1)

    while np.sum(image[-1]) == 0:
        image = image[:-1]

    while np.sum(image[:, -1]) == 0:
        image = np.delete(image, -1, 1)

    rows, cols = image.shape

    # Resize the resulting image into a 28 x 28 image
    # while maintaining the aspect ratio
    if rows > cols:
        factor = 28.0 / rows
        rows = 28
        cols = int(round(cols * factor))
    else:
        factor = 28.0 / cols
        cols = 28
        rows = int(round(rows * factor))
    image = cv2.resize(image, (cols, rows))

    # Pad the 28 x 28 so that it is now 32 x 32
    column_padding = (int(math.ceil((32-cols)/2.0)),
                      int(math.ceil((32-cols)/2.0)))
    row_padding = (int(math.ceil((32-rows)/2.0)),
                   int(math.ceil((32-rows)/2.0)))
    image = np.lib.pad(image,
                       (row_padding, column_padding),
                       'constant')

    # Center the character in this new image
    x_shift, y_shift = get_best_shift(image)
    shifted = shift(image, x_shift, y_shift)
    image = shifted

    return image


def preprocess_images(directory):
    """
    DESCRIPTION: Preprocess all of the images in the dataset directory.
    This assumes that the directory has already been restructured with
    nist_restructure.py.
    INPUT: Dataset path
    OUTPUT: Processed dataset
    """
    # Start a timer
    start = time.time()

    # From the parent directory, we go into both the test and training sets
    dataset_path = pathlib.Path(directory)
    # For each set, we want to go into each label
    for dat_set in dataset_path.iterdir():
        for char in dat_set.iterdir():
            # For every image in the label directory, resize and shift
            for image in char.iterdir():
                # Read in the image as a grayscale image
                img = cv2.imread(str(image), 0)
                width, height = img.shape

                if width != WIDTH or height != HEIGHT:
                    processed_img = preprocess(img)
                    # Overwrite old image
                    cv2.imwrite(str(image), processed_img)
                    print('{} resized'.format(str(image)))
                else:
                    print('{} already resized'.format(str(image)))
    print('> Completion Time: {}'.format(time.time() - start))


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", help="path to dataset", required=True)
    ARGS = vars(AP.parse_args())

    preprocess_images(ARGS['directory'])
