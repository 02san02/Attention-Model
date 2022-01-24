# Importing the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Dropout, LeakyReLU, GlobalAveragePooling2D    
from tensorflow.keras import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json, clone_model, load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

# Model Definition
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

input1 = Input(shape=input_shape)

# Resuidal block BN -> relu -> conv -> bn -> relu -> conv
def res_block(x, filters, stem=False):
    bn1 = BatchNormalization()(x)
    act1 = Activation('relu')(bn1)
    conv1 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2), padding='same', 
                   kernel_initializer=glorot_uniform(seed=0))(act1)
    print('conv1.shape', conv1.shape)
    
    bn2 = BatchNormalization()(conv1)
    act2 = Activation('relu')(bn2)
    
    conv2 = Conv2D(filters=filters, kernel_size=(5, 5), strides=(1, 1), padding='same', 
                  kernel_initializer=glorot_uniform(seed=0))(act2)
    print('conv2.shape', conv2.shape)
    
    residual = Conv2D(1, (1, 1), strides=(1, 1))(conv2)
    
    x = Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2), padding='same', 
                   kernel_initializer=glorot_uniform(seed=0))(x)
    print('x.shape', x.shape)

    out = Add()([x, residual])
    
    return out

# Combining resuidal blocks into a network
res1 = res_block(input1, 32, True)
print('---------block 1 end-----------')
res2 = res_block(res1, 64)
print('---------block 2 end-----------')
res3 = res_block(res2, 128)
print('---------block 3 end-----------')
res4 = res_block(res3, 256)
print('---------block 4 end-----------')

# Classifier block
act1 = Activation('relu')(res4)
flatten1 = GlobalAveragePooling2D()(act1)
#flatten1 = Flatten()(act1)
dense1 = Dense(1024)(flatten1)
act2 = Activation('relu')(dense1)
output1 = Dense(2, activation="softmax")(act2)

model2 = Model(inputs=input1, outputs=output1)
