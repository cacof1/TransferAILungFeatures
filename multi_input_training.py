# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 08:17:42 2021

@author: zhuoy
"""


import pandas as pd
from time import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
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
import tensorflow_transform as tft
from tensorflow.keras import backend as K
from tensorflow .random import set_seed
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
import pandas as pd
from multi_input_model import *

basepath = "C:/Users/zhuoy/Note/Radiomics/"
outcome_path = basepath+"NCT00533949-D1-Dataset.csv"
outcome = pd.read_csv(outcome_path)


prediction_tags = ['rt_compliance_physician', 'rt_compliance_ptv90', 'received_conc_chemo', 'received_conc_cetuximab', 'received_cons_chemo', 'received_cons_cetuximab', 'conc_cetux_score_overall',
                   'conc_chemo_score_overall', 'conc_cetux_score_dose', 'conc_chemo_score_dose', 'conc_cetux_score_delays', 'conc_chemo_score_delays', 'cons_cetux_score_overall',
                   'cons_chemo_score_overall', 'cons_cetux_score_dose', 'cons_chemo_score_dose', 'cons_cetux_score_delays', 'cons_chemo_score_delays', 'grade3_esophagitis', 'grade3_pneumonitis',
                   'grade3_pulmonary', 'grade3_toxicity', 'grade5_toxicity', 'include_in_late_ae', 'survival_status', 'survival_months', 'cod', 'local_failure', 'local_failure_months', 'distant_failure',
                   'distant_failure_months', 'progression_free_survival','progression_free_survival_months', 'lost_to_followup']
    
prediction_tag = 'survival_months'
outcome        = outcome[outcome[prediction_tag].notna()] ## Drop those that we haven't (NaN)
ids_outcome    = outcome['patid']

## Load the preprocessed data (image + dose)
database       = np.load("database_masked.npz", allow_pickle = True)
patient_data   = np.array(database['data'])    
ids_patient    = np.array(database['patid'])        

ids_common     = sorted(list(set(ids_outcome.values).intersection(ids_patient)))
outcome        = outcome[outcome['patid'].isin(ids_common)]
outcome        = outcome.reset_index()

X_init         = patient_data[np.isin(ids_patient,ids_common)]
y_init         = outcome.loc[:,prediction_tag]
y_init         = (y_init>24).astype('int16')

np.savez("survival.npz",data = y_init,patid =ids_common)  

X_imgs = np.load('features_masked_Xception.npz', allow_pickle = True)
features = np.array(X_imgs['data'])
X_imgs = features[np.isin(ids_patient,ids_common)]
X_doses = X_init[:,:,:,:,1]
y = np.load('survival.npz', allow_pickle = True)
df_y= pd.DataFrame.from_dict({item: y[item] for item in y.files})

X1 = X_imgs
X2 = X_doses

print(X1.shape)
print(X2.shape)


X1_train, X1_test, y_train, y_test = train_test_split(X1, df_y, test_size=0.3, random_state=12)


print('Shape of X1_train:{}'.format(X1_train.shape))
print('Shape of X1_test:{}'.format(X1_test.shape))
print('Shape of y_train:{}'.format(y_train.shape))
print('Shape of y_test:{}'.format(y_test.shape))

restid = list(set(df_y.index).difference(set(y_train.index)))

X2_train = []

for i in y_train.index:
    X2_train.append(X2[i])
    
X2_test = []

for i in restid:
    X2_test.append(X2[i])
    
X2_train = np.array(X2_train)
X2_test = np.array(X2_test)
    
print('Shape of X2_train:{}'.format(X2_train.shape))
print('Shape of X2_test:{}'.format(X2_test.shape))

y_train = y_train['data']
y_test = y_test['data']

print('Shape of y_train:{}'.format(y_train.shape))
print('Shape of y_test:{}'.format(y_test.shape))

#%%
X1_train = tf.cast(X1_train, tf.bfloat16)
X1_test  = tf.cast(X1_test, tf.bfloat16)
X2_train = tf.cast(X2_train, tf.bfloat16)
X2_test  = tf.cast(X2_test, tf.bfloat16)
y_train = tf.cast(y_train, tf.bfloat16)
y_test  = tf.cast(y_test, tf.bfloat16)

size1 = {'weight':X1.shape[1],
         'height':X1.shape[2],
         }

size2 = {'weight':X2.shape[-2],
         'height':X2.shape[-1],
         'depth':X2.shape[1]
         }


model = ct_dose_model(size1,size2)

model.summary()

lr = 0.01
momentum = 0.5
batch_size = 10
print('lr = ' + str(lr))
print('momentum = ' + str(momentum))
print('batch size = ' + str(batch_size))

####### Dataset generator -- for efficient dataset loading
#train_dataset = My_Custom_Generator(X_train, y_train, batch_size)
#test_dataset = My_Custom_Generator(X_test, y_test, batch_size)
#train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
#test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

####### Define callbacks and learning rates.
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("best_classifier.h5", monitor="val_auc", mode='max', save_best_only=True)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode='max', patience=50)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( lr, decay_steps=100000, decay_rate=0.96, staircase=True)

####### Compilation step. Needed to initialize the model, pick loss, optimizer, etc..
model.compile(loss="binary_crossentropy",  optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), metrics=tf.keras.metrics.AUC(name='auc'))
#%%

###### Training
history = model.fit(
    x=[X1_train, X2_train], y=y_train,
	validation_data=([X1_test, X2_test], y_test),
    epochs = 100,
    #shuffle = True,
    verbose = 2,
    callbacks=[checkpoint_cb, early_stopping_cb],)


#%%
def plot_history(histories, key):

    plt.figure(figsize=(6,5))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    
plot_history([('baseline', history)],'auc')
plot_history([('baseline', history)],'loss')
plt.show()
#%%
##### Testing on holdout
model = load_model('best_classifier.h5')
scores = model.evaluate([X1_test, X2_test], y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
y_pred = model.predict([X1_test, X2_test])
y_classes = y_pred.argmax(axis=-1)

fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr,label='2D CNN Dose/Img - AUC:%.3f'%roc_auc)
print("2D 2-Channel",roc_auc)
plt.legend(frameon=False)
plt.plot([0,1],[0,1],'r--o')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

