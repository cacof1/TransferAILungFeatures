# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 07:56:52 2021

@author: zhuoy
"""


import pandas as pd
from time import time
from tensorflow.keras import applications, optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPool3D, MaxPooling2D,PReLU, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.random import set_seed
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from numpy.random import seed
import pandas as pd

def feature_dose_model(size1,size2):
    #inputs include image and dose
    inputA = tf.keras.Input((size1['weight'], size1['height'], 1))
    inputB = tf.keras.Input((size2['depth'],size2['weight'], size2['height'],1))
    
    x = layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu")(inputA)
    x = layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2)(x)
    x = layers.PReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2)(x)
    x = layers.PReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x  = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(units=512, activation="relu")(x)
    #x = layers.Dropout(0.5)(x)

    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    x = tf.keras.Model(inputs=inputA, outputs=x, name="image_model")
    
    y = layers.Conv3D(filters=16, kernel_size=[3,3,3], activation="relu")(inputB)
    y = layers.MaxPool3D(pool_size=2,padding='same')(y)
    y = layers.PReLU()(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)

    y = layers.Conv3D(filters=32, kernel_size=[3,3,3], activation="relu")(y)
    y = layers.MaxPool3D(pool_size=[2,2,2],padding='same')(y)
    y = layers.PReLU()(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)
    
    y = layers.Conv3D(filters=160, kernel_size=2, activation="relu")(y)
    y = layers.MaxPool3D(pool_size=2,padding='same')(y)
    y = layers.PReLU()(y)
    y = layers.BatchNormalization()(y)

    #y = layers.Conv3D(filters=256, kernel_size=2, activation="relu")(y)
    #y = layers.MaxPool3D(pool_size=2,padding='same')(y)
    #y = layers.BatchNormalization()(y)

    y  = layers.GlobalAveragePooling3D()(y)
    y = layers.Dense(units=256, activation="relu")(y)
    y = layers.Dropout(0.5)(y)
    
    y = tf.keras.Model(inputs=inputB, outputs=y, name="dose_model")
    
    combined = layers.Concatenate()([x.output, y.output])
    
    z = layers.Dense(units=128,  activation="relu")(combined)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(units=64,  activation="relu")(z)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(units=1, activation="sigmoid")(z)
    
    model = Model(inputs=[x.input, y.input], outputs=z)
    
    return model
    
def feature_dose_clinical_model(size1,size2,size3):
    #inputs include image/dose/clinical
    
    inputA = tf.keras.Input((size1['weight'], size1['height'], 1))
    inputB = tf.keras.Input((size2['depth'],size2['weight'], size2['height'],1))
    inputC = tf.keras.Input((size3))
    
    x = layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu")(inputA)
    x = layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2)(x)
    x = layers.PReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2)(x)
    x = layers.PReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x  = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(units=512, activation="relu")(x)
    #x = layers.Dropout(0.5)(x)

    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    x = tf.keras.Model(inputs=inputA, outputs=x, name="image_model")
    
    y = layers.Conv3D(filters=16, kernel_size=[3,3,3], activation="relu")(inputB)
    y = layers.MaxPool3D(pool_size=2,padding='same')(y)
    y = layers.PReLU()(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)

    y = layers.Conv3D(filters=32, kernel_size=[3,3,3], activation="relu")(y)
    y = layers.MaxPool3D(pool_size=[2,2,2],padding='same')(y)
    y = layers.PReLU()(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.5)(y)
    
    #y = layers.Conv3D(filters=160, kernel_size=2, activation="relu")(y)
    #y = layers.MaxPool3D(pool_size=2,padding='same')(y)
    #y = layers.PReLU()(y)
    #y = layers.BatchNormalization()(y)
    #y = layers.Dropout(0.5)(y)

    y  = layers.GlobalAveragePooling3D()(y)
    y = layers.Dense(units=256, activation="relu")(y)
    y = layers.Dropout(0.5)(y)
    
    y = tf.keras.Model(inputs=inputB, outputs=y, name="dose_model")
    
    z = layers.Dense(units=32,  activation="relu")(inputC)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(units=32,  activation="relu")(z)
    z = layers.Dropout(0.5)(z)
    z = tf.keras.Model(inputs=inputC, outputs=z, name="clinical_model")
    
    combined = layers.Concatenate()([x.output, y.output, z.output])
    
    combined = layers.Dense(units=256,  activation="relu")(combined)
    combined = layers.Dropout(0.5)(combined)
    combined = layers.Dense(units=128,  activation="relu")(combined)
    combined = layers.Dropout(0.5)(combined)
    combined = layers.Dense(units=1, activation="sigmoid")(combined)
    
    model = Model(inputs=[x.input, y.input,z.input], outputs=combined)
    
    return model
    
    
def feature_feature_clinical_model(size1,size3):
    #inputs include image/dose/clinical
    
    inputA = tf.keras.Input((size1['weight'], size1['height'], size1['channel']))
    inputC = tf.keras.Input((size3))
    
    x = layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu")(inputA)
    x = layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2)(x)
    x = layers.PReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2)(x)
    x = layers.PReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    
    x  = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dense(units=512, activation="relu")(x)
    #x = layers.Dropout(0.5)(x)

    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    
    x = tf.keras.Model(inputs=inputA, outputs=x, name="ct_dose_model")
    
    z = layers.Dense(units=32,  activation="relu")(inputC)
    z = layers.Dropout(0.5)(z)
    z = layers.Dense(units=32,  activation="relu")(z)
    z = layers.Dropout(0.5)(z)
    
    z = tf.keras.Model(inputs=inputC, outputs=z, name="clinical_model")
    
    combined = layers.Concatenate()([x.output, z.output])
    
    combined = layers.Dense(units=128,  activation="relu")(combined)
    combined = layers.Dropout(0.5)(combined)
    combined = layers.Dense(units=128,  activation="relu")(combined)
    combined = layers.Dropout(0.5)(combined)
    combined = layers.Dense(units=1, activation="sigmoid")(combined)
    
    model = Model(inputs=[x.input,z.input], outputs=combined)
    
    return model
    
