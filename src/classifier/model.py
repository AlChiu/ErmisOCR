"""
Keras implementation of a character classifier
"""
import time
import os
import warnings
import keras
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = 3
warnings.filterwarnings('ignore')

# Hyperparameters
DROPOUT = 0.5
STRIDES = 2
HEIGHT = 128
WIDTH = 128
KERNEL_SIZE_1 = [5, 5]
KERNEL_SIZE_2 = [3, 3]
POOL_SIZE = (2, 2)
CONV1_FILTERS = 64
CONV2_FILTERS = 128
CONV3_FILTERS = 256
CONV4_FILTERS = 512
CONV5_FILTERS = 512
NEURON_UNITS = 100


def build_model():
    """Build the neural network model using keras style initialization"""
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(
        input_shape=(HEIGHT, WIDTH, 1),
        filters=CONV1_FILTERS,
        kernel_size=KERNEL_SIZE_1,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 1
    model.add(MaxPooling2D(
        pool_size=POOL_SIZE,
        strides=STRIDES))

    # Convolutional Layer 2
    model.add(Conv2D(
        filters=CONV2_FILTERS,
        kernel_size=KERNEL_SIZE_1,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 2
    model.add(MaxPooling2D(
        pool_size=POOL_SIZE,
        strides=STRIDES))

    # Convolutional Layer 3
    model.add(Conv2D(
        filters=CONV3_FILTERS,
        kernel_size=KERNEL_SIZE_2,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 3
    model.add(MaxPooling2D(
        pool_size=POOL_SIZE,
        strides=STRIDES))

    # Convolutional Layer 4
    model.add(Conv2D(
        filters=CONV4_FILTERS,
        kernel_size=KERNEL_SIZE_2,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 4
    model.add(MaxPooling2D(
        pool_size=POOL_SIZE,
        strides=STRIDES))

    # Convolutional Layer 5
    model.add(Conv2D(
        filters=CONV5_FILTERS,
        kernel_size=KERNEL_SIZE_2,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 5
    model.add(MaxPooling2D(
        pool_size=POOL_SIZE,
        strides=STRIDES))

    # Flatten the output of convolution for the classification
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(
        units=NEURON_UNITS,
        activation='relu'))
    model.add(Dropout(rate=DROPOUT/2))

    # Dense Layer 2
    model.add(Dense(
        units=NEURON_UNITS,
        activation='relu'))
    model.add(Dropout(rate=DROPOUT))

    # Classification Layer / Dense Layer 3
    model.add(Dense(
        units=62,
        activation='softmax'))

    start = time.time()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print('> Compilation Time: {}'.format(time.time() - start))
    return model
