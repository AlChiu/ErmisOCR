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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def build_model_1():
    """
    Model #1: ADADELTA Optimizer
    Conv 1 - 64 3x3 kernels with padding
    Max Pool 1 - 2x2 kernel stride 2
    Conv 2 - 128 3x3 kernels with padding
    Max Pool 2 - 2x2 kernel stride 2
    Conv 3 - 256 1x1 kernels with padding
    Max Pool 3 - 2x2 kernel stride 2
    FC 1 - 100 neurons with .25 dropout chance
    FC 2 - 100 neuraons with .5 dropout chance
    FC 3 - 62 neurons softmax classifier
    """
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(
        input_shape=(28, 28, 1),
        filters=64,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 1
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # Convolutional Layer 2
    model.add(Conv2D(
        filters=128,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 2
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # Convolutional Layer 3
    model.add(Conv2D(
        filters=256,
        kernel_size=1,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 3
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # Flatten the output of convolution for the classification
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(
        units=100,
        activation='relu'))
    model.add(Dropout(rate=.25))

    # Dense Layer 2
    model.add(Dense(
        units=100,
        activation='relu'))
    model.add(Dropout(rate=.5))

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


def build_model_2():
    """
    Model #2: ADADELTA Optimizer
    Conv 1 - 32 5x5 kernels with padding
    Max Pool 1 - 2x2 kernel stride 2
    Conv 2 - 64 5x5 kernels with padding
    Max Pool 2 - 2x2 kernel stride 2
    FC 1 - 1024 neurons with .4 dropout chance
    FC 2 - 62 neurons softmax classifier
    """
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(
        input_shape=(28, 28, 1),
        filters=32,
        kernel_size=5,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 1
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # Convolutional Layer 2
    model.add(Conv2D(
        filters=64,
        kernel_size=5,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 2
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # Flatten the output of convolution for the classification
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(
        units=1024,
        activation='relu'))
    model.add(Dropout(rate=.4))

    # Classification Layer
    model.add(Dense(
        units=62,
        activation='softmax'))

    start = time.time()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print('> Compilation Time: {}'.format(time.time() - start))
    return model


def build_model_3():
    """
    Model #3: ADADELTA Optimizer
    Conv 1 - 32 3x3 kernels
    Conv 2 - 32 3x3 kernels
    Max Pool 1 - 2x2 kernel stride 2
    Conv 3 - 64 3x3 kernels
    Conv 3 - 64 3x3 kernels
    Max Pool 2 - 2x2 kernel stride 2
    FC 1 - 512 neurons with .2 dropout chance
    FC 2 - 62 neurons softmax classifier
    """
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(
        input_shape=(28, 28, 1),
        filters=32,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Convolutional Layer 2
    model.add(Conv2D(
        filters=32,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 1
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # Convolutional Layer 3
    model.add(Conv2D(
        filters=64,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Convolutional Layer 4
    model.add(Conv2D(
        filters=64,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 2
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # Flatten the output of convolution for the classification
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(
        units=512,
        activation='relu'))
    BatchNormalization()
    model.add(Dropout(rate=.2))

    # Classification Layer
    model.add(Dense(
        units=62,
        activation='softmax'))

    start = time.time()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print('> Compilation Time: {}'.format(time.time() - start))
    return model


def build_model_4():
    """
    Model #4 (VGGNet5): ADADELTA Optimizer
    Conv 1 = 32 3x3 kernels
    Conv 2 = 32 3x3 kernels
    Max Pool 1 = 2x2 stride 2
    Conv 3 = 64 3x3 kernels
    Conv 4 = 64 3x3 kernels
    Max Pool 2 = 2x2 stride 2
    Conv 5 = 128 3x3 kernels
    Conv 6 = 128 3x3 kernels
    Conv 7 = 128 3x3 kernels
    Max Pool 3 = 2x2 stride 2
    Conv 8 = 256 3x3 kernels
    Conv 9 = 256 3x3 kernels
    Conv 10 = 256 3x3 kernels
    Max Pool 4 = 2x2 stride 2
    FC 1 - 512 neurons with .3 dropout chance
    FC 2 - 512 neuraons with .5 dropout chance
    FC 3 - 62 neurons softmax classifier
    """
    model = Sequential()

    # BLOCK 1 #
    # Convolutional Layer 1
    model.add(Conv2D(
        input_shape=(28, 28, 1),
        filters=32,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Convolutional Layer 2
    model.add(Conv2D(
        filters=32,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 1
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # BLOCK 2 #
    # Convolutional Layer 3
    model.add(Conv2D(
        filters=64,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Convolutional Layer 4
    model.add(Conv2D(
        filters=64,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 2
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # BLOCK 3 #
    # Convolutional Layer 5
    model.add(Conv2D(
        filters=128,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Convolutional Layer 6
    model.add(Conv2D(
        filters=128,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Convolutional Layer 7
    model.add(Conv2D(
        filters=128,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 3
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # BLOCK 4 #
    # Convolutional Layer 8
    model.add(Conv2D(
        filters=256,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Convolutional Layer 9
    model.add(Conv2D(
        filters=256,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Convolutional Layer 10
    model.add(Conv2D(
        filters=256,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 4
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # BLOCK 5
    # Flatten the output of convolution for the classification
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(
        units=512,
        activation='relu'))
    model.add(Dropout(rate=.3))

    # Dense Layer 2
    model.add(Dense(
        units=512,
        activation='relu'))
    model.add(Dropout(rate=.5))

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


def build_model_5():
    """
    Model #5: ADADELTA Optimizer
    Conv 1 - 64 3x3 kernels
    Max Pool 1 - 2x2 kernel stride 2
    Conv 2 - 128 3x3 kernels
    Max Pool 2 - 2x2 kernel stride 2
    Conv 3 - 256 1x1 kernels
    Max Pool 3 - 2x2 kernel stride 2
    FC 1 - 100 neurons with .25 dropout chance
    FC 2 - 100 neuraons with .5 dropout chance
    FC 3 - 62 neurons softmax classifier
    """
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(
        input_shape=(28, 28, 1),
        filters=64,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 1
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # Convolutional Layer 2
    model.add(Conv2D(
        filters=128,
        kernel_size=3,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 2
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # Convolutional Layer 3
    model.add(Conv2D(
        filters=256,
        kernel_size=1,
        padding="same"))
    BatchNormalization(axis=-1)
    model.add(Activation('relu'))

    # Pooling Layer 3
    model.add(MaxPooling2D(
        pool_size=2,
        strides=2))

    # Flatten the output of convolution for the classification
    model.add(Flatten())

    # Dense Layer 1
    model.add(Dense(
        units=100,
        activation='relu'))
    model.add(Dropout(rate=.25))

    # Dense Layer 2
    model.add(Dense(
        units=100,
        activation='relu'))
    model.add(Dropout(rate=.5))

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
