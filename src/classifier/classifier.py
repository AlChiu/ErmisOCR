"""classifier.py
Define the classifier class and the preprocessing method
"""
import os
import glob
import math
import json
import cv2
from scipy import ndimage
import numpy as np
from keras.models import load_model
from src.detector import char_detect_segment as det_seg


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
    image = cv2.imread(image, 0)
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

    while np.sum(image[: -1]) == 0:
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
    image = cv2.resize(shifted, (32, 32))

    return image


def concatenate_list_data(char_list):
    """
    DESCRIPTION: List concatenation of characters to produce words.
    INPUT: Translated character list
    OUTPUT: A single element that represents a word
    """
    result = ''
    for element in char_list:
        result += str(element)
    return result


class Classifier:
    """
    DESCRIPTION: Defines the classifier for image prediction.
    Needs a trained neural model to function properly.
    """
    def __init__(self, model, lab_dict):
        """
        Initialize the character classifier with the
        specified trained neural network model and character
        dictionary.
        """
        if model is not None:
            self.model = load_model(model)
        else:
            raise ValueError('A trained model must be provided.')

        if lab_dict is not None:
            with open(lab_dict) as file:
                self.labels = json.load(file)
        else:
            raise ValueError('A label dictionary is required')

    def classify_one(self, image_path):
        """
        DESCRIPTION: Predict and classify a character image.
        INTPUT: Path to character image.
        OUTPUT: Character class label of the prediction.
        """
        if self.model is not None:
            image = preprocess(image_path)
            image = image.reshape(-1, 32, 32, 1)
            outp = self.model.predict(image)[0]

            top_idx = outp.argmax(axis=-1)

            character = self.labels.get(str(top_idx))
            print(character)

            return character
        else:
            raise ValueError('A trained model must be provided')

    def classify_many(self, image_dirc):
        """
        DESCRIPTION: Classify many characters of a word
        using the classify_one function.
        INPUT: Directory of word image files
        OUTPUT: A string representing a single word
        """
        word = []
        if self.model is not None:
            for image in sorted(glob.iglob(image_dirc + 'char_*'),
                                key=det_seg.numerical_sort):
                word.append(self.classify_one(image))
                os.remove(image)

            c_word = concatenate_list_data(word)
            return c_word
