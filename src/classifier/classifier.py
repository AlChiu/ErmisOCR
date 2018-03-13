"""
Classifier class definition
"""
import os
import glob
import argparse
import cv2
from keras.models import load_model
from src.detector import char_detect_segment as det_seg

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


def preprocess(image_path):
    """
    Function to preprocess the test image the same
    way that produces the MNIST 28x28 image, just
    at 224 x 224.
    """
    image = cv2.imread(image_path, 0)
    (_, image) = cv2.threshold(image, 0, 255,
                               cv2.THRESH_BINARY_INV |
                               cv2.THRESH_OTSU)
    image = cv2.resize(image, (224, 224))
    image = cv2.normalize(image, None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F)
    return image


def concatenate_list_data(char_list):
    """
    Perform a list concatenation of characters to produce
    words.
    """
    result = ''
    for element in char_list:
        result += str(element)
    return result


class Classifier:
    """
    Defines the classifier for image prediction.
    Needs a trained neural model to function properly.
    """
    def __init__(self, model):
        """
        Initialize the character classifier with the
        specified trained neural network model.
        """
        if model is not None:
            self.model = load_model(model)
        else:
            raise ValueError('A trained model must be provided.')

    def classify_one(self, image_path):
        """
        Attempt to predict and classify. Return an array of the
        top 5 resulting predictions with confidence levels.
        """
        # First, preprocess the image so that it is similar to
        # MNIST images at 224 x 224
        if self.model is not None:
            image = preprocess(image_path)
            image = image.reshape(-1, 224, 224, 1)
            outp = self.model.predict(image)[0]

            top_idx = (-outp).argsort()[:1]

            for char in top_idx:
                character = CHAR_LABELS.get(char)

            return character

        else:
            raise ValueError('A trained model must be provided')

    def classify_many(self, image_dirc):
        """
        Used to classify the many characters in a word image
        """
        word = []
        if self.model is not None:
            for image in sorted(glob.iglob(image_dirc + 'char_*'),
                                key=det_seg.numerical_sort):
                word.append(self.classify_one(image))
                os.remove(image)

            c_word = concatenate_list_data(word)
            return c_word
