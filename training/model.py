"""
Keras implementation of a character classifier
"""
import os
import warnings
import keras
from keras.layers import Input, concatenate
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential, Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


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
                  optimizer=keras.optimizers.rmsprop(lr=0.001,
                                                     rho=0.9,
                                                     epsilon=None,
                                                     decay=0.0),
                  metrics=['accuracy'])
    return model


def squeeze_net(classes=62, inputs=(224, 224, 1)):
    """
    Keras implementation of Squeeze Net
    """
    input_img = Input(shape=inputs)
    conv1 = Conv2D(96, (7, 7), strides=(2, 2),
                   padding='same', activation='relu')(input_img)
    max_pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv1)

    # Fire Module 1
    fire1_sq = Conv2D(16, (1, 1), activation='relu', padding='same')(max_pool1)
    fire1_e1 = Conv2D(64, (1, 1), activation='relu', padding='same')(fire1_sq)
    fire1_e3 = Conv2D(64, (3, 3), activation='relu', padding='same')(fire1_sq)
    merge1 = concatenate([fire1_e1, fire1_e3], axis=3)

    # Fire Module 2
    fire2_sq = Conv2D(16, (1, 1), activation='relu', padding='same')(merge1)
    fire2_e1 = Conv2D(64, (1, 1), activation='relu', padding='same')(fire2_sq)
    fire2_e3 = Conv2D(64, (3, 3), activation='relu', padding='same')(fire2_sq)
    merge2 = concatenate([fire2_e1, fire2_e3], axis=3)

    # Fire Module 3
    fire3_sq = Conv2D(32, (1, 1), activation='relu', padding='same')(merge2)
    fire3_e1 = Conv2D(128, (1, 1), activation='relu', padding='same')(fire3_sq)
    fire3_e3 = Conv2D(128, (3, 3), activation='relu', padding='same')(fire3_sq)
    merge3 = concatenate([fire3_e1, fire3_e3], axis=3)

    # Fire Module 4
    fire4_sq = Conv2D(32, (1, 1), activation='relu', padding='same')(merge3)
    fire4_e1 = Conv2D(128, (1, 1), activation='relu', padding='same')(fire4_sq)
    fire4_e3 = Conv2D(128, (3, 3), activation='relu', padding='same')(fire4_sq)
    merge4 = concatenate([fire4_e1, fire4_e3], axis=3)

    # Fire Module 5
    fire5_sq = Conv2D(32, (1, 1), activation='relu', padding='same')(merge4)
    fire5_e1 = Conv2D(128, (1, 1), activation='relu', padding='same')(fire5_sq)
    fire5_e3 = Conv2D(128, (3, 3), activation='relu', padding='same')(fire5_sq)
    merge5 = concatenate([fire5_e1, fire5_e3], axis=3)

    # Fire Module 6
    fire6_sq = Conv2D(48, (1, 1), activation='relu', padding='same')(merge5)
    fire6_e1 = Conv2D(192, (1, 1), activation='relu', padding='same')(fire6_sq)
    fire6_e3 = Conv2D(192, (3, 3), activation='relu', padding='same')(fire6_sq)
    merge6 = concatenate([fire6_e1, fire6_e3], axis=3)

    # Fire Module 7
    fire7_sq = Conv2D(48, (1, 1), activation='relu', padding='same')(merge6)
    fire7_e1 = Conv2D(192, (1, 1), activation='relu', padding='same')(fire7_sq)
    fire7_e3 = Conv2D(192, (3, 3), activation='relu', padding='same')(fire7_sq)
    merge7 = concatenate([fire7_e1, fire7_e3], axis=3)

    # Fire Module 8
    fire8_sq = Conv2D(64, (1, 1), activation='relu', padding='same')(merge7)
    fire8_e1 = Conv2D(256, (1, 1), activation='relu', padding='same')(fire8_sq)
    fire8_e3 = Conv2D(256, (3, 3), activation='relu', padding='same')(fire8_sq)
    merge8 = concatenate([fire8_e1, fire8_e3], axis=3)

    # Fire Module 9
    fire9_sq = Conv2D(64, (1, 1), activation='relu', padding='same')(merge8)
    fire9_e1 = Conv2D(256, (1, 1), activation='relu', padding='same')(fire9_sq)
    fire9_e3 = Conv2D(256, (3, 3), activation='relu', padding='same')(fire9_sq)
    merge9 = concatenate([fire9_e1, fire9_e3], axis=3)

    # Classify
    fire9_drop = Dropout(0.5)(merge9)
    conv10 = Conv2D(classes, (1, 1), padding='valid')(fire9_drop)
    glo_avg = GlobalAveragePooling2D()(conv10)
    softmax = Activation('softmax')(glo_avg)

    model = Model(inputs=input_img, outputs=softmax)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=1e-3,
                                                 momentum=0.9,
                                                 decay=1e-4,
                                                 nesterov=True),
                  metrics=['accuracy'])
    return model
