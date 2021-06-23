# -*- coding: utf-8 -*-
"""
Created on Thu May 27 05:53:26 2021

@author: zhuoy
"""

from tensorflow.keras.layers import concatenate
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras import Input
from tensorflow.keras import Model
import numpy as np
import pandas as pd

ratio = 1
or_image_width = 160
or_image_height = 160
channels = 1

image_width = int(or_image_width / ratio)
image_height = int(or_image_height / ratio)

img_input = Input(shape=(image_width,image_height,1))
img_conc = concatenate([img_input, img_input, img_input], axis=-1) 

model_name = 'NASNetLarge'

baseModel = NASNetLarge(weights="imagenet", 
                  include_top=False,
                  pooling = 'avg',
                  input_tensor=img_conc)
 
print("[INFO] summary for base model...")
print(baseModel.summary())

## Load the preprocessed data (image + dose)
database       = np.load("database_masked.npz", allow_pickle = True)
patient_data   = np.array(database['data'])    
ids_patient    = np.array(database['patid'])        

X_init         = patient_data

#%%Extract feature
#intermediate_layer_model = Model(inputs=baseModel.input, outputs=baseModel.get_layer('block4_pool').output)
num_of_patients = X_init.shape[0]
num_of_slices = X_init.shape[1]

X_imgs = X_init[:,:,:,:,0]
i = 0
features = []
#intermediate_features = []
for i in range(num_of_slices):
    X_img = X_imgs[:,i,:,:]
    X_img = np.expand_dims(X_img, axis=-1)
    feature = baseModel.predict(X_img)
    #intermediate_feature = intermediate_layer_model.predict(X_img)
    features.append(feature)
    #intermediate_features.append(intermediate_feature)
    i = i+1
    print('{}/{} slices completed'.format(i,num_of_slices))
    

features = np.array(features)
height = features.shape[-1]

features = features.reshape(num_of_patients,num_of_slices,height)
#intermediate_features = np.array(intermediate_features)


X_doses = X_init[:,:,:,:,1]

i = 0
dose_features = []

for i in range(num_of_slices):
    X_dose = X_doses[:,i,:,:]
    X_dose = np.expand_dims(X_dose, axis=-1)
    dose_feature = baseModel.predict(X_dose)
    dose_features.append(dose_feature)
    i = i+1
    print('{}/{} slices completed'.format(i,num_of_slices))

dose_features = np.array(dose_features)
height = dose_features.shape[-1]

dose_features = dose_features.reshape(num_of_patients,num_of_slices,height)

np.savez("features_masked_{}.npz".format(model_name),data = features,patid =ids_patient)              
np.savez("dose_masked.npz",data = X_doses, patid =ids_patient)                                        #Save dose without transferring
np.savez("dose_features_masked_{}.npz".format(model_name),data = dose_features,patid =ids_patient)

