"""
Train_classifier will load the training and testing
data, build the neural network graph, then train. It will
return and save the model as a .h5 file
"""
import os
import time
import random
import argparse
import numpy as np
import keras
from keras import backend as K
import model
import convert_data

HEIGHT = 32
WIDTH = 32


def create_feed_data(directory, train_pkl_no, test_pkl_no):
    """
    Function will go to the directory and unpickle the
    two provided files. It will then return four lists
    that will be used for the neural network
    INPUT: 1) Directory of the pickle files
           2) Training pickle subset no.
           3) Test pickle subset no.
    OUTPUT: 1) train_img - list of training image arrays
            2) train_lbl - list of training image labels
            3) test_img - list of test image arrays
            4) test_lbl - list of test image lables
    """
    # Create the lists
    train_img = []
    train_lbl = []
    test_img = []
    test_lbl = []

    try:
        # Unpickle the pickles
        train_data = convert_data.load_data(directory,
                                            train_pkl_no,
                                            "training")
        test_data = convert_data.load_data(directory,
                                           test_pkl_no,
                                           "testing")

        # Shuffle the loaded lists in place
        random.shuffle(train_data, random.random)
        random.shuffle(test_data, random.random)

        # Create the four new lists
        for line in train_data:
            train_img.append(line[3])
            train_lbl.append(line[4])
            # train_lbl.append(keras.utils.to_categorical(line[2], 62))

        for line in test_data:
            test_img.append(line[3])
            test_lbl.append(line[4])
            # test_lbl.append(keras.utils.to_categorical(line[2], 62))

        # Reshape and create one-hot vectors for labels
        print('{} training images and {} testing images'.format(len(train_img),
                                                                len(test_img)))

        train_img = np.array(train_img)
        train_lbl = keras.utils.to_categorical(train_lbl, 62)
        test_img = np.array(test_img)
        test_lbl = keras.utils.to_categorical(test_lbl, 62)

        train_img = train_img.reshape(train_img.shape[0], HEIGHT, WIDTH, 1)
        test_img = test_img.reshape(test_img.shape[0], HEIGHT, WIDTH, 1)

    except FileNotFoundError:
        print('Pickle files were not found in {}'.format(directory))

    return train_img, train_lbl, test_img, test_lbl


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", help="path to dataset")
    AP.add_argument("-t", "--text", help="path to text_file")
    ARGS = vars(AP.parse_args())

    #############################
    # Feed Test
    #############################
    create_feed_data(ARGS['directory'], 0, 0)
