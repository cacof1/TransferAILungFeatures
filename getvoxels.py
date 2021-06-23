# -*- coding: utf-8 -*-
"""
Created on Thu May 27 02:48:29 2021

@author: zhuoy
"""
import sys, os,glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import stats
from scipy import ndimage
basepath = '/lustre/projects/ParticleCT/OutcomePrediction/Study/RTOG0617/nrrd_volumes'
#database = {}
patientdata  = np.zeros((1,20,160,160,2),dtype=np.float32)
database     = np.zeros((1,20,160,160,2),dtype=np.float32)
id_list      = []
NPatient = 0
for dirName, subdirList, fileList in os.walk(basepath):
    if("CT_unmasked.nrrd" in fileList and "dose.nrrd" in fileList):
        patient_id = dirName.split("/")[-1]
        img_path  = os.sep.join([dirName, "CT_unmasked.nrrd"])
        dose_path = os.sep.join([dirName, "dose.nrrd"])        
        
        img  = sitk.ReadImage(img_path)
        dose = sitk.ReadImage(dose_path)
        
        img  = sitk.GetArrayFromImage(img).astype(np.float32)
        dose = sitk.GetArrayFromImage(dose).astype(np.float32)
        
        shape = list(img.shape)
        
        ids  = np.where(img<-100)
        ids2  = np.where(img>-100)
        ## Little trick to not mix image values at zero and cancer HU at zero
        img += img+1024
        img[ids] = 0

        cm_z, cm_y, cm_x = list(map(int,ndimage.measurements.center_of_mass(dose))) 
        #img[ids2] -= 1000

        img  = img[cm_z-25:cm_z+25 , cm_y-128:cm_y+128 , cm_x-128:cm_x+128 ] ## 256x256x50
        dose = dose[cm_z-25:cm_z+25 , cm_y-128:cm_y+128 , cm_x-128:cm_x+128 ] ## 256x256x50


        img[np.where(img>0)]   = stats.zscore(img[np.where(img>0)])
        dose[np.where(dose>0)] = stats.zscore(dose[np.where(dose>0)])
        
        patientdata[0,:,:,:,0] = img ## Cheap and ugly but does the job
        patientdata[0,:,:,:,1] = dose
        if(NPatient==0):
            database[0] = patientdata
        else:
            database = np.append(database,patientdata,axis=0)
        print(database.shape)
        print(patient_id)
        id_list.append(patient_id)
        NPatient = NPatient +1
        print(NPatient)
        if(NPatient>400): break
    else:
        continue        


np.savez("database.npz",data = database,patid =id_list)





