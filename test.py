import tensorflow as tf
from keras.models import Model
from keras.layers import (Input, Convolution2D, Lambda, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout, Activation)
from keras.utils import plot_model

# This layer normalizes the outputs of a neural network layer.
# Similar to Batch Normalization layer but deprecated in use.
def lrn(input, radius=5, alpha=0.0005, beta=0.75, name='LRN', bias=1.0):
        return tf.nn.local_response_normalization(input, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name) 

def cnn_m_2048(name, input_shape, keep_prob_1, keep_prob_2=1., num_classes=None, get_feature_vector=True):
    
    inputs = Input(shape=input_shape, name='inputs_{}'.format(name))
    # BLOCK 1
    conv1 = Convolution2D(96, (7, 7), strides=2, activation='relu', name='conv1_{}'.format(name))(inputs)   
    lrn1 = Lambda(lrn)(conv1)
    pool1 = MaxPooling2D((3,3), strides=(2,2), name='pool1_{}'.format(name))(lrn1)
    
    # BLOCK 2
    padding2 = ZeroPadding2D(padding=(1,1))(pool1)
    conv2 = Convolution2D(256, (5, 5), strides=2,activation='relu', name='conv2_{}'.format(name))(padding2)
    lrn2 = Lambda(lrn)(conv2)
    pool2 = MaxPooling2D((3,3), strides=(2,2),name='pool2_{}'.format(name))(lrn2)
    
    # BLOCK 3
    padding3 = ZeroPadding2D(padding=(1,1))(pool2)
    conv3 = Convolution2D(512, (3, 3), strides=(1,1),activation='relu', name='conv3_{}'.format(name))(padding3)
    
    # BLOCK 4
    padding4 = ZeroPadding2D(padding=(1,1))(conv3)
    conv4 = Convolution2D(512, (3, 3), strides=(1,1),activation='relu', name='conv4_{}'.format(name))(padding4)
    
    # BLOCK 5
    padding5 = ZeroPadding2D(padding=(1,1))(conv4)
    conv5 = Convolution2D(512, (3, 3), strides=(1,1), activation='relu', name='conv5_{}'.format(name))(padding5)
    pool5 = MaxPooling2D((3,3), strides=(2,2), name='pool5_{}'.format(name))(conv5)

    flatten = Flatten()(pool5)
    
    # MULTILAYER PERCEPTRON (CLASSIFIER)
    dense1 = Dense(4096, activation='relu', name='fc6_{}'.format(name))(flatten)
    dropout1 = Dropout(keep_prob_1)(dense1)
    dense2 = Dense(2048, activation='relu', name='fc7_{}'.format(name))(dropout1)
        
    dropout2 = Dropout(keep_prob_2)(dense2)
    logits = Dense(num_classes, name='fc8_{}'.format(name))(dropout2)
    softmax = Activation('softmax')(logits)
    return Model(inputs, softmax)

image_size = (224, 224)
stan_dogs_model = cnn_m_2048(name='test', input_shape=image_size + (3,), keep_prob_1=0.5, num_classes=4)
plot_model(stan_dogs_model, show_shapes=True)