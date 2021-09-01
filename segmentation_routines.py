# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:37:51 2021

v2.0 CT processing code
This file contains all the rountines for segmenting into pores and part mask.

@author: rw1816
"""
import cv2
import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage import measure
import os.path
import pickle

def isolate_pores(pores, mask, img_params):
    
    #add a block which can deal with the dual areas??
    labels = measure.label(pores, background=255)
    props = measure.regionprops(labels)
    areas = [prop.area for prop in props]
    pores_clean = remove_small_objects(labels, min_size=img_params['min pore'])  
    kernel=np.ones((img_params['edge trim'], img_params['edge trim']))
    mask=cv2.erode(mask.astype('uint8'), kernel)
    pores_clean[mask!=1] = 0 #i.e. anything within kernel of the edge is not a pore
    pores_clean[pores_clean==areas.index(max(areas))+1] = 0
    pores_clean[pores_clean!=0] = 1
    pores_out=np.zeros(pores_clean.shape).astype('uint8')
    pores_out[4:-4,4:-4]=pores_clean[4:-4,4:-4]

    return pores_out

def pores_from_mask(bin, mask, img_params):
    pores = bin.copy()
    kernel = np.ones((img_params['kernel 2'],img_params['kernel 2']), np.uint8)
    pores = cv2.morphologyEx(pores, cv2.MORPH_CLOSE, kernel)
    pores = isolate_pores(pores, mask, img_params)
    
    return pores

def mask_and_pores(img, img_params, adapt=True, gauss=True):
    
    kernel = np.ones((img_params['kernel 1'], img_params['kernel 1']), np.uint8)
    
    if gauss:
        img=cv2.GaussianBlur(img,(img_params['gauss kernel'],img_params['gauss kernel']),0)
    
    if adapt:
        bin = cv2.adaptiveThreshold((img/256).astype('uint8'), 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY, 15, 8)
        _, mask =cv2.threshold(img, img_params['mask thresh'], 255, cv2.THRESH_BINARY)

    else:

        _, mask = cv2.threshold(img, img_params['mask thresh'], 255, cv2.THRESH_BINARY)
        
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    labels = measure.label(mask)
    mask = remove_small_holes(labels, area_threshold=img_params['mask max hole'], 
                              connectivity=1, in_place=True)    #remove anything under 5pi
    pores =pores_from_mask(bin, mask, img_params)
    
    return mask.astype('uint8'), pores

def create_img_dict(save_path):
    
    if os.path.isfile(save_path):
        with open(save_path, 'rb') as p:
            img_params=pickle.load(p)
    
    else:
        # define image processing parameters
        img_params = {"mask thresh" : 10000,
                      "min pore" : 3,
                      "mask max hole" : 20,
                      "kernel 1" : 7,
                      "kernel 2" : 5,
                      "edge trim" : 25,
                      "encoding" : 'uint16',
                      "gauss kernel" : 3
                      }
        with open(save_path, 'wb') as p:
            pickle.dump(img_params, p)
        
    return img_params