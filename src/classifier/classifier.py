"""
Classifier class definition
"""
import os
import glob
import json
import cv2
from keras.models import load_model
from src.detector import char_detect_segment as det_seg


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
    def __init__(self, model, lab_dict):
        """
        Initialize the character classifier with the
        specified trained neural network model.
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
        Attempt to predict and classify. Return an array of the
        top 5 resulting predictions with confidence levels.
        """
        # First, preprocess the image so that it is similar to
        # MNIST images at 224 x 224
        if self.model is not None:
            image = preprocess(image_path)
            image = image.reshape(-1, 224, 224, 1)
            outp = self.model.predict(image)[0]

            top_idx = outp.argmax(axis=-1)

            character = self.labels.get(top_idx)

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
