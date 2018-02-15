"""
Classifier class definition
"""
import argparse
import numpy as np
import cv2
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
            self.model = load_model(model)
        elif model_path is not None:
            self.model = load_model(model_path)
        else:
            raise ValueError('Either model or model_path must be given')

    def preprocess(self, image_path):
        """
        Function to preprocess the test image the same
        way that produces the MNIST 28x28 image, just
        at 224 x 224.
        """
        image = cv2.imread(image_path, 0)
        image = convert_data.resize_shift(image)
        image = cv2.normalize(image, None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX)
        image = cv2.resize(image, (224, 224))
        image = np.asarray(image, dtype=np.float32)
        cv2.imshow("letter", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image

    def classify(self, image_path):
        """
        Attempt to predict and classify. Return an array of the
        top 5 resulting predictions with confidence levels.
        """
        # First, preprocess the image so that it is similar to
        # MNIST images at 224 x 224
        image = self.preprocess(image_path)
        image = image.reshape(-1, 224, 224, 1)
        outp = np.array(self.model.predict(image)[0])

        top5_idx = outp.argsort()[-5:]
        top5_chars = []
        for char in top5_idx:
            top5_chars.append([CHAR_LABELS.get(char), outp[char]])

        top5_chars = sorted(top5_chars, key=lambda x: x[1], reverse=True)

        print(top5_chars)


if __name__ == "__main__":
    # Build the argument to load in the image
    AP = argparse.ArgumentParser()
    AP.add_argument("-i", "--image", help="singular image path")
    ARGS = vars(AP.parse_args())

    # Create the classifier
    classifier = Classifier(model='model_62char.h5')
    classifier.classify(ARGS['image'])
