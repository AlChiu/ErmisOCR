"""
Keras implementation of a character classifier
"""
import os
import warnings
import keras
from keras.layers.core import Activation, Dropout, Flatten, Dense
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

    # shape = (1, 1, int(1024 * alpha))

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
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     input_shape=(width, height, 1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv2 Block
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # FC1 Block
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(400, activation='relu'))

    # Classifier Block
    model.add(Dense(classes, activation='softmax'))

    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.rmsprop(lr=0.001),
                  metrics=['accuracy'])
    return model


def lenet(classes, height, width):
    """
    LeNet-5 Based classifier
    """
    inp_shape = (width, height, 1)
    model = Sequential()

    # Conv1-Input Layer
    model.add(Conv2D(12, (5, 5),
                     activation='relu',
                     input_shape=inp_shape,
                     kernel_initializer='he_normal'))

    # Pooling Layer 1
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv2 Layer
    model.add(Conv2D(25, (5, 5),
                     activation='relu',
                     kernel_initializer='he_normal'))

    # Pooling Layer 2
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Fully Connected 1
    model.add(Flatten())
    model.add(Dense(180, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))

    # Fully Connected 2
    model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
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


def vgg_esque(classes, height, width):
    """
    VGG based with bottlenecking
    """
    inp_shape = (width, height, 1)
    model = Sequential()

    # Conv Block 1
    model.add(Conv2D(128, (3, 3), activation='relu',
                     input_shape=inp_shape,
                     padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Conv Block 2
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu',
                     padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Fully Connected 1
    model.add(Flatten())
    model.add(Dense(750, activation='relu'))
    model.add(Dropout(0.5))

    # Fully Connected 2
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))

    # Classifier
    model.add(Dense(classes, activation='softmax'))

    # Compile
    model.compile(loss='categorical_crossentropy',
                  optimizer='adagrad',
                  metrics=['accuracy'])
    return model
