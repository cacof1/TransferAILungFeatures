from tensorflow.keras.layers import concatenate
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Input
from tensorflow.keras import Model
import numpy as np
import pandas as pd

'''

def normalization(x):
    #x:numpy array
    x = x.astype('float32')
    print('Original Range of Pixel Value:{}-{}'.format(x.min(),x.max()))
    x /= 255.0
    print('Normalized Range of Pixel Value:{}-{}'.format(x.min(),x.max()))
    return x

def local_centering(x):
    #x:numpy array
    x = x.astype('float32')
    means = x.mean(axis=(0,1),dtype='float64')
    print('Original Means:R:{},G:{},B:{}'.format(means[0],means[1],means[2]))
    x -= means
    means = x.mean(axis=(0,1),dtype='float64')
    print('Centerred Means:R:{},G:{},B:{}'.format(means[0],means[1],means[2]))
    return x
'''
ratio = 1
or_image_width = 256
or_image_height = 256
channels = 1

image_width = int(or_image_width / ratio)
image_height = int(or_image_height / ratio)

img_input = Input(shape=(image_width,image_height,1))
img_conc = concatenate([img_input, img_input, img_input], axis=-1) 

baseModel = VGG16(weights="imagenet", 
                  include_top=False,
                  input_tensor=img_conc)
 
print("[INFO] summary for base model...")
print(baseModel.summary())

#%%Import data

basepath = "/lustre/projects/ParticleCT/OutcomePrediction/Study/RTOG0617/"
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
database       = np.load("/lustre/projects/ParticleCT/OutcomePrediction/TransferAI/database_unmasked.npz", allow_pickle = True)
patient_data   = np.array(database['data'])    
ids_patient    = np.array(database['patid'])        

ids_common     = sorted(list(set(ids_outcome.values).intersection(ids_patient)))
outcome        = outcome[outcome['patid'].isin(ids_common)]
outcome        = outcome.reset_index()

X_init         = patient_data[np.isin(ids_patient,ids_common)]
y_init         = outcome.loc[:,prediction_tag]
#y_init         = (y_init>24).astype('int16') ## Survival at 2 years as a boolean

#%%Extract feature
#intermediate_layer_model = Model(inputs=baseModel.input, outputs=baseModel.get_layer('block4_pool').output)

X_imgs = X_init[:,:,:,:,0]
i = 0
features = []
#intermediate_features = []
for i in range(60):
    X_img = X_imgs[:,i,:,:]
    X_img = np.expand_dims(X_img, axis=-1)
    feature = baseModel.predict(X_img)
    #intermediate_feature = intermediate_layer_model.predict(X_img)
    features.append(feature)
    #intermediate_features.append(intermediate_feature)
    i = i+1

features = np.array(features)
features = features.reshape(325,60,5,5,512)
#intermediate_features = np.array(intermediate_features)

X_doses = X_init[:,:,:,:,1]
i = 0
dose_features = []

for i in range(60):
    X_dose = X_doses[:,i,:,:]
    X_dose = np.expand_dims(X_dose, axis=-1)
    dose_feature = baseModel.predict(X_dose)
    dose_features.append(dose_feature)
    i = i+1

dose_features = np.array(dose_features)
dose_features = dose_features.reshape(325,60,5,5,512)

np.savez("features_unmasked.npz",data = features,patid =ids_common)
np.savez("dose_features_unmasked.npz",data = dose_features,patid =ids_common)
np.savez("survival.npz",data = y_init,patid =ids_common)