"""classifier.py
Define the classifier class and the preprocessing method
"""
import os
import glob
import json
import cv2
from keras.models import load_model
from src.detector import char_detect_segment as det_seg


def preprocess(image_path):
    """
    DESCRIPTION: Preprocess the image the same way that we preprocessed the
    dataset.
    INPUT: Image path
    OUTPUT: Processed image resized to 32 x 32
    """
    image = cv2.imread(image_path, 0)
    (_, image) = cv2.threshold(image, 0, 255,
                               cv2.THRESH_BINARY_INV |
                               cv2.THRESH_OTSU)
    image = cv2.resize(image, (32, 32))
    image = cv2.normalize(image, None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_32F)
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
