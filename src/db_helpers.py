#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:41:23 2018

@author: owen
"""
from glob import glob
import numpy as np
import os
from src.simgraph import SimGraph, from_file

def dir_to_graphlist(indir,ftype='*.simg'):
    
    files = glob(indir+ftype) #get all npy files in directory
    
    glist = [] #initialize list
    for f in files: #iterate over each file in director
        fname = os.path.splitext(os.path.basename(f))[0] #file name
        glist.append((fname,from_file(f))) #load sim graph and append to list
    
    return glist
        
def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def simg_to_carvalho_names(simg_str):
    temp_str = simg_str.rsplit('_',1)[0]
    mask_str = temp_str + '.png'
    im_str = rreplace(simg_str,'_','.',1)
    return im_str, mask_str

def simg_to_korus_names(simg_str):
    mask_str = simg_str.replace('_TIF','.PNG')
    im_str = simg_str.replace('_TIF','.TIF')
    return im_str, mask_str

def simg_to_columbia_names(simg_str,edge_mask_end='_edgemask_3.jpg'):
    mask_str = simg_str.replace('_tif',edge_mask_end)
    im_str = simg_str.replace('_tif','.tif')
    return im_str, mask_str
