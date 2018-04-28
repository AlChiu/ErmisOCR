"""
Keras implementation of a character classifier
"""
import os
import warnings
import keras
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def simple_1(classes, height, width):
    """
    First model
    Conv1 has 12 5x5 kernels
    Pool1 uses a 2x2 kernel
    Conv2 has 25 5x5 kernels
    Pool2 uses a 2x2 kernel
    Dense1 has 180 neurons with 0.5 dropout
    Dense2 has 100 neurons with 0.5 dropout
    Dense3 is the classifier
    """
    inp_shape = (width, height, 1)
    model = Sequential()

    # Conv1-Input Layer
    model.add(Conv2D(12, (5, 5),
                     activation='relu',
                     input_shape=inp_shape,
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())

    # Pooling Layer 1
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv2 Layer
    model.add(Conv2D(25, (5, 5),
                     activation='relu',
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # Pooling Layer 2
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Fully Connected 1
    model.add(Flatten())
    model.add(Dense(180, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Fully Connected 2
    model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Classifier
    model.add(Dense(classes,
                    activation='softmax',
                    kernel_initializer='he_normal'))

    # Compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def simple_2(classes, height, width):
    """
    Second Model
    Conv1 has 32 3x3 kernels
    Pool1 uses 2x2 kernel
    Conv2 has 64 3x3 kernels
    Pool2 uses 2x2 kernel
    Conv3 has 128 1x1 kernels
    Dense1 has 400 neurons with 0.5 dropout
    Dense2 has 200 neurons with 0.5 dropout
    Dense3 is the classifier
    """
    inp_shape = (width, height, 1)
    model = Sequential()

    # Conv1-Input Layer
    model.add(Conv2D(32, (3, 3),
                     activation='relu',
                     input_shape=inp_shape,
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # Pooling Layer 1
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv2 Layer
    model.add(Conv2D(64, (3, 3),
                     activation='relu',
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # Pooling Layer 2
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv3 Layer
    model.add(Conv2D(128, (1, 1), activation='relu',
                     kernel_initializer='he_normal'))

    # Fully Connected 1
    model.add(Flatten())
    model.add(Dense(400, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Fully Connected 2
    model.add(Dense(200, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Classifier
    model.add(Dense(classes,
                    activation='softmax',
                    kernel_initializer='he_normal'))

    # Compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def simple_3(classes, height, width):
    """
    Third model
    Conv1 has 16 5x5 kernels
    Pool1 uses 2x2 kernel
    Conv2 has 32 3x3 kernels
    Pool2 uses 2x2 kernel
    Conv3 has 64 1x1 kernel
    Dense1 has 300 neurons with 0.5 Dropout
    Dense2 has 150 neurons with 0.5 Dropout
    Dense3 is the classifier
    """
    inp_shape = (width, height, 1)
    model = Sequential()

    # Conv1-Input Layer
    model.add(Conv2D(16, (5, 5),
                     activation='relu',
                     input_shape=inp_shape,
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # Pooling Layer 1
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv2 Layer
    model.add(Conv2D(32, (3, 3),
                     activation='relu',
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # Pooling Layer 2
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv3 Layer
    model.add(Conv2D(64, (1, 1), activation='relu',
                     kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    # Fully Connected 1
    model.add(Flatten())
    model.add(Dense(300, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Fully Connected 2
    model.add(Dense(150, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Classifier
    model.add(Dense(classes,
                    activation='softmax',
                    kernel_initializer='he_normal'))

    # Compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model
