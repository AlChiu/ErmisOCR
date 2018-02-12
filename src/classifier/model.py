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


def VGG16_Char():
    """
    Model #5: VGGNet16 -3
    """
    model = Sequential()

    # BLOCK 1
    model.add(Conv2D(64, (3, 3), input_shape=(224, 224, 1),
              padding='same', name='blk1_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', name='blk1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='blk1_pool'))

    # BLOCK 2
    model.add(Conv2D(128, (3, 3), padding='same', name='blk2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', name='blk2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='blk2_pool'))

    # BLOCK 3
    model.add(Conv2D(256, (3, 3), padding='same', name='blk3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', name='blk3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', name='blk3_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='blk3_pool'))

    # BLOCK 4
    model.add(Conv2D(512, (3, 3), padding='same', name='blk4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', name='blk4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', name='blk4_conv3'))
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
    model.add(Dense(units=2048, name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=2048, name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(units=62, name='classifier'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    start = time.time()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=1e-3,
                                                 momentum=0.9,
                                                 decay=1e-6,
                                                 nesterov=True),
                  metrics=['accuracy'])
    print('> Compilation Time: {}'.format(time.time() - start))
    return model
