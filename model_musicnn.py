"""
VGG-like backbone
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Conv2D, MaxPool2D, Dropout, Flatten

INPUT_SIZE = (187, 96)
OUTPUT_SIZE = 50

SR = 16000
FFT_HOP = 256
FFT_SIZE = 512
N_MELS = 96


def vgg_keras(num_filters=128, return_feature_model=False):
    ''' VGG-audio backbone. '''
    X = Input(INPUT_SIZE)
    input_layer = Lambda(lambda x: tf.expand_dims(x, -1))(X)
    bn_input = BatchNormalization(axis=-1, name='bn_input')(input_layer)

    conv1 = Conv2D(num_filters, (3, 3),
                    padding='same',
                    activation="relu",
                    name='1CNN')(bn_input)
    bn_conv1 = BatchNormalization(axis=-1, name='bn_conv1')(conv1)
    pool1 = MaxPool2D(pool_size=[4, 1], strides=[2, 2],name='pool1')(bn_conv1)

    do_pool1 = Dropout(0.25)(pool1)
    conv2 = Conv2D(num_filters, (3, 3),
                    padding='same',
                    activation="relu",
                    name='2CNN')(do_pool1)
    bn_conv2 = BatchNormalization(axis=-1, name='bn_conv2')(conv2)
    pool2 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], name='pool2')(bn_conv2)

    do_pool2 = Dropout(0.25)(pool2)
    conv3 = Conv2D(num_filters, (3, 3),
                    padding='same',
                    activation="relu",
                    name='3CNN')(do_pool2)

    bn_conv3 = BatchNormalization(axis=-1, name='bn_conv3')(conv3)
    pool3 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], name='pool3')(bn_conv3)

    do_pool3 = Dropout(0.25)(pool3)
    conv4 = Conv2D(num_filters, (3, 3),
                    padding='same',
                    activation="relu",
                    name='4CNN')(do_pool3)

    bn_conv4 = BatchNormalization(axis=-1, name='bn_conv4')(conv4)
    pool4 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], name='pool4')(bn_conv4)

    do_pool4 = Dropout(0.25)(pool4)
    conv5 = Conv2D(num_filters, (3, 3),
                    padding='same',
                    activation="relu",
                    name='5CNN')(do_pool4)

    bn_conv5 = BatchNormalization(axis=-1, name='bn_conv5')(conv5)
    pool5 = MaxPool2D(pool_size=[4, 4], strides=[4, 4], name='pool5')(bn_conv5)

    flat_pool5 = Flatten()(pool5)
    do_pool5 = Dropout(0.5)(flat_pool5)
    output = Dense(OUTPUT_SIZE, activation = 'sigmoid', name='output')(do_pool5)

    model = tf.keras.Model(X, output)
    if not return_feature_model:
        return model

    # i figured the two first layer may be too noisy and big and inefficient anyway
    features = tf.keras.Model(X, [pool3, pool4, pool5, output])
    return model, features


def var_vgg_keras(num_filters=128, output_layers = (1 << 5) - 1):
    ''' The same but allowing to have input of any size. '''
    X = Input((None,) + INPUT_SIZE[1:] )
    input_layer = Lambda(lambda x: tf.expand_dims(x, -1))(X)
    bn_input = BatchNormalization(axis=-1, name='bn_input')(input_layer)

    conv1 = Conv2D(num_filters, (3, 3),
                    padding='same',
                    activation="relu",
                    name='1CNN')(bn_input)
    bn_conv1 = BatchNormalization(axis=-1, name='bn_conv1')(conv1)
    pool1 = MaxPool2D(pool_size=[4, 1], strides=[2, 2],name='pool1')(bn_conv1)

    do_pool1 = Dropout(0.25)(pool1)
    conv2 = Conv2D(num_filters, (3, 3),
                    padding='same',
                    activation="relu",
                    name='2CNN')(do_pool1)
    bn_conv2 = BatchNormalization(axis=-1, name='bn_conv2')(conv2)
    pool2 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], name='pool2')(bn_conv2)

    do_pool2 = Dropout(0.25)(pool2)
    conv3 = Conv2D(num_filters, (3, 3),
                    padding='same',
                    activation="relu",
                    name='3CNN')(do_pool2)

    bn_conv3 = BatchNormalization(axis=-1, name='bn_conv3')(conv3)
    pool3 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], name='pool3')(bn_conv3)

    do_pool3 = Dropout(0.25)(pool3)
    conv4 = Conv2D(num_filters, (3, 3),
                    padding='same',
                    activation="relu",
                    name='4CNN')(do_pool3)

    bn_conv4 = BatchNormalization(axis=-1, name='bn_conv4')(conv4)
    pool4 = MaxPool2D(pool_size=[2, 2], strides=[2, 2], name='pool4')(bn_conv4)

    do_pool4 = Dropout(0.25)(pool4)
    conv5 = Conv2D(num_filters, (3, 3),
                    padding='same',
                    activation="relu",
                    name='5CNN')(do_pool4)

    bn_conv5 = BatchNormalization(axis=-1, name='bn_conv5')(conv5)
    pool5 = MaxPool2D(pool_size=[4, 4], strides=[4, 4], name='pool5')(bn_conv5)

    pooling_layers = [pool1, pool2, pool3, pool4, pool5]
    needed_layers = []
    for i, l in enumerate(pooling_layers):
        if (1 << i) & output_layers > 0:
            needed_layers.append(l)
    print(needed_layers)
    model = tf.keras.Model(X, needed_layers)

    return model


