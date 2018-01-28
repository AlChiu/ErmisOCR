"""
Train_classifier will load the training and testing
data, build the neural network graph, then train. It will
return and save the model as a .h5 file
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import random
import argparse
import numpy as np
import keras
from keras.models import load_model
from keras.callbacks import History
import model
import convert_data

HISTORY = History()
HEIGHT = 32
WIDTH = 32
DIVISION = 10
BATCH_SIZE = 200
EPOCHS = 50


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
        print('> {} training images loaded'.format(len(train_img)))
        print('> {} testing images loaded'.format(len(test_img)))

        train_img = np.array(train_img)
        train_lbl = keras.utils.to_categorical(train_lbl, 62)
        test_img = np.array(test_img)
        test_lbl = keras.utils.to_categorical(test_lbl, 62)

        # Reshape the image arrays into numpy arrays for the network
        train_img = train_img.reshape(train_img.shape[0], HEIGHT, WIDTH, 1)
        test_img = test_img.reshape(test_img.shape[0], HEIGHT, WIDTH, 1)

    except FileNotFoundError:
        print('> Pickle files were not found in {}'.format(directory))

    return train_img, train_lbl, test_img, test_lbl


def build_model():
    """Build the Keras neural network"""
    print("> Building Keras neural network...")
    network_model = model.build_model()
    return network_model


def fit_model(model, train_img, train_lbl, test_img, test_lbl, name):
    """
    Train the neural network with batch size of 200, 100 epochs,
    and print out training progress.
    """
    model.fit(
        train_img,
        train_lbl,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(test_img, test_lbl),
        callbacks=[HISTORY]
    )

    score = model.evaluate(test_img, test_lbl, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuaracy: ', score[1])

    model.save(name)
    return model


def plot(model, setting, count):
    """Plot the accuracy and loss of training"""
    fig = setting + '_' + str(count) + '_graph.png'
    if setting == "accuracy":
        plt.plot(model.History['acc'])
        plt.plot(model.History['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
    elif setting == "loss":
        plt.plot(model.History['loss'])
        plt.plot(model.History['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')

    plt.xlabel('Epoch')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.savefig(fig)


if __name__ == "__main__":
    # Build up the argument to bring in an image
    AP = argparse.ArgumentParser()
    AP.add_argument("-d", "--directory", help="path to dataset")
    AP.add_argument("-t", "--text", help="path to text_file")
    ARGS = vars(AP.parse_args())

    #############################
    # Feed Test
    #############################
    MODEL_NAME = 'model_62char.h5'
    DIRECTORY = ARGS['directory']
    MODEL_DIRECTORY = os.path.join(DIRECTORY, MODEL_NAME)

    COUNT = 0
    while COUNT < DIVISION:
        train_img, train_lbl, test_img, test_lbl = create_feed_data(DIRECTORY,
                                                                    COUNT,
                                                                    COUNT)
        MODEL_FILE = Path(MODEL_DIRECTORY)
        if MODEL_FILE.is_file():
            char_model = load_model(MODEL_DIRECTORY)
            history = fit_model(char_model,
                                train_img,
                                train_lbl,
                                test_img,
                                test_lbl,
                                MODEL_DIRECTORY)
            # plot(history, "accuracy", COUNT)
            # plot(history, "loss", COUNT)
        else:
            char_model = build_model()
            history = fit_model(char_model,
                                train_img,
                                train_lbl,
                                test_img,
                                test_lbl,
                                MODEL_DIRECTORY)
            # plot(history, "accuracy", COUNT)
            # plot(history, "loss", COUNT)
        COUNT += 1
