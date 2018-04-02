"""
Keras implementation of a character classifier
"""
import os
import warnings
import keras
from keras.layers.core import Activation, Dropout, Reshape, Flatten, Dense
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def mobile_net(classes, height, width, alpha=1):
    """
    Google's Mobile Net modified to use ELU activation.
    """
    model = Sequential()

    model.add(Conv2D(int(32*alpha), (3, 3), strides=(2, 2),
                     input_shape=(width, height, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConv2D(int(64*alpha), (3, 3), strides=(1, 1),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConv2D(int(128*alpha), (3, 3), strides=(2, 2),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConv2D(int(128*alpha), (3, 3), strides=(1, 1),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConv2D(int(256*alpha), (3, 3), strides=(2, 2),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConv2D(int(256*alpha), (3, 3), strides=(1, 1),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConv2D(int(512*alpha), (3, 3), strides=(2, 2),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    for _ in range(5):
        model.add(SeparableConv2D(int(512*alpha), (3, 3), strides=(1, 1),
                                  padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('elu'))

    model.add(SeparableConv2D(int(1024*alpha), (3, 3), strides=(2, 2),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(SeparableConv2D(int(1024*alpha), (1, 1), strides=(2, 2),
                              padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    shape = (1, 1, int(1024 * alpha))

    model.add(GlobalAveragePooling2D())
    # model.add(Reshape(shape))
    # model.add(Dropout(rate=1e-3))
    # model.add(Conv2D(classes, (1, 1), padding='same'))
    # model.add(Activation('softmax'))
    # model.add(Reshape((classes,)))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.rmsprop(lr=0.001,
                                                     rho=0.9,
                                                     epsilon=None,
                                                     decay=0.0),
                  metrics=['accuracy'])
    return model


def simple_net(classes, height, width):
    """
    Simple Toy ConvNet for testing
    """
    model = Sequential()

    # Conv1 Block
    model.add(64, (3, 3), activation='relu', padding='same',
              input_shape=(width, height, 1))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv2 Block
    model.add(128, (3, 3), activation='relu', padding='same')
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # FC1 Block
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(200), activation='relu')

    # Classifier Block
    model.add(Dense(classes, activation='softmax'))

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.rmsprop(lr=0.001),
                  metrics=['accuracy'])
    return model
