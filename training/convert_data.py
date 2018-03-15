"""
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
    (_, image) = cv2.threshold(image, 0, 255,
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
    Resize all of the images in the dataset directory. This assumes that
    the directory has already been restructured.
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
                # Grab the filename
                name = image.parts[-3]
                # Read in the image as a grayscale image
                img = cv2.imread(str(image), 0)
                # Grab its initial dimensions
                width, height = img.shape

                if width != WIDTH or height != HEIGHT:
                    # Resize the image to 224 x 224
                    resized_img = resize_shift(img)
                    # Save the image with the same filename
                    cv2.imwrite(str(image), resized_img)
                    # Print whether the image was resized or not
                    print('{} resized'.format(name))
                else:
                    print('{} already has the correct dimensions'.format(name))
    print('> Completion Time: {}'.format(time.time() - start))


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", help="path to dataset")
    ARGS = vars(AP.parse_args())

    resize_images(ARGS['directory'])
