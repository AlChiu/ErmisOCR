"""
Keras implementation of a character classifier
"""
import os
import warnings
import keras
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def VGG16_Char():
    """
    Model #5: VGGNet16 -3
    """
    model = Sequential()

    # BLOCK 1
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 1),
              padding='same', name='blk1_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', name='blk1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='blk1_pool'))

    # BLOCK 2
    model.add(Conv2D(64, (3, 3), padding='same', name='blk2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', name='blk2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='blk2_pool'))

    # BLOCK 3
    model.add(Conv2D(128, (3, 3), padding='same', name='blk3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', name='blk3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', name='blk3_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='blk3_pool'))

    # BLOCK 4
    model.add(Conv2D(256, (3, 3), padding='same', name='blk4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', name='blk4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', name='blk4_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='blk4_pool'))

    # BLOCK 5
    model.add(Conv2D(512, (3, 3), padding='same', name='blk5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', name='blk5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', name='blk5_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='blk5_pool'))

    # CLASSIFICATION
    model.add(Flatten(name='flatten'))
    model.add(Dense(units=1024, name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=1024, name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=62, name='classifier'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=1e-3,
                                                 momentum=0.9,
                                                 decay=1e-4,
                                                 nesterov=True),
                  metrics=['accuracy'])
    return model


def mobile_net(alpha=1, classes=62):
    """
    Google's Mobile Net modified to use ELU activation.
    """
    model = Sequential()

    model.add(Conv2D(int(32*alpha), (3, 3), strides=(2, 2),
              input_shape=(224, 224, 1), padding='same'))
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
    model.add(Reshape(shape))
    model.add(Dropout(rate=1e-3))
    model.add(Conv2D(classes, (1, 1), padding='same'))
    model.add(Activation('softmax'))
    model.add(Reshape((classes,)))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=1e-3,
                                                 momentum=0.9,
                                                 decay=1e-4,
                                                 nesterov=True),
                  metrics=['accuracy'])
    return model
