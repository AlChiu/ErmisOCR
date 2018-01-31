"""
Classifier class definition
"""
import math
import cv2
import numpy as np
from scipy import ndimage
from keras.models import load_model
import convert_data

CHAR_LABELS = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'a',
    11: 'b',
    12: 'c',
    13: 'd',
    14: 'e',
    15: 'f',
    16: 'g',
    17: 'h',
    18: 'i',
    19: 'j',
    20: 'k',
    21: 'l',
    22: 'm',
    23: 'n',
    24: 'o',
    25: 'p',
    26: 'q',
    27: 'r',
    28: 's',
    29: 't',
    30: 'u',
    31: 'v',
    32: 'w',
    33: 'x',
    34: 'y',
    35: 'z',
    36: 'A',
    37: 'B',
    38: 'C',
    39: 'D',
    40: 'E',
    41: 'F',
    42: 'G',
    43: 'H',
    44: 'I',
    45: 'J',
    46: 'K',
    47: 'L',
    48: 'M',
    49: 'N',
    50: 'O',
    51: 'P',
    52: 'Q',
    53: 'R',
    54: 'S',
    55: 'T',
    56: 'U',
    57: 'V',
    58: 'W',
    59: 'X',
    60: 'Y',
    61: 'Z'
}


class Classifier:
    """
    Defines the classifier for image prediction.
    Needs a trained neural model to function properly.
    """
    def __init__(self, model=None, model_path=None):
        """
        Initialize the character classifier with the
        specified trained neural network model.
        """
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = load_model(model_path)
        else:
            raise ValueError('Either model or model_path must be given')

    def preprocess(self, image_path):
        """
        Function to preprocess the test image the same
        way that produces the MNIST 28x28 image.
        """
        print('preprocessing')

    def classify(self, image_path):
        """
        Attempt to predict and classify. Return an array of the
        top 5 resulting predictions with confidence levels.
        """
        try:
            image_array = convert_data.convert_to_pixel_array(image_path)
            image_array = np.array(image_array)
            input_image = np.array([image_array])
            input_image = input_image.reshape(input_image.size[0], 32, 32, 1)

            prediction = self.model.predict(input_image)[0]
            prediction = np.array(prediction)

            top5_array = prediction.argsort()[-5:]

            print(top5_array)

            return top5_array
        except FileNotFoundError:
            print('Cannot find the image file.')
