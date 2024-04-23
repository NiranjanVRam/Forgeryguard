#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 16:02:35 2019

@author: owen
"""

import numpy as np
import os 
from sklearn.metrics import average_precision_score 
from sklearn.metrics import precision_recall_curve
import pickle
import matplotlib.pyplot as plt

class DetectionResult():
    #has:    
        #authentic
        #spliced
        #result_type
        #label
        #notes
    #does:
        #calculate average precision
        #calculate roc
        #calculate pd at pfa
        #plot roc to axis
        #save
        def __init__(self,authentic,spliced,result_type=None,label=None,parameters=None):
            self.auth = authentic 
            self.splc = spliced
            self.rtype = result_type
            self.label = label
            self.params = parameters
        
        def calc_roc(self,T=None):
            #calculate pfa,pd for roc curve
            vpfa,vpd = roc(self.auth,self.splc,T)
            self.vpfa = vpfa
            self.vpd = vpd
            return vpfa,vpd
        
        def calc_map(self):
            #calculate mean average precision
            meanAP = calc_map(self.auth,self.splc)
            return meanAP
        
        def plot_roc(self,ax=None):
            if ax is None:
                fig, ax = plt.subplots(1)
            
            #check if roc is calculated
            if not hasattr(self,'vpfa'):
                self.calc_roc()
            
            handle = ax.plot(self.vpfa,self.vpd,label=self.label)
            return ax, handle
        
        def calc_pd_at_pfa(self,pfa):
            if not hasattr(self,'vpfa'):
                self.calc_roc()
            val = np.interp(pfa,np.flip(self.vpfa),np.flip(self.vpd))
            return val

        def calc_roc_auc(self):
            if not hasattr(self,'vpfa'):
                self.calc_roc()
            auc = np.trapz(np.flip(self.vpd),np.flip(self.vpfa))
            return auc
            
        def save(self,name):
            with open(name,'wb') as f:
                pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
                
                               
def detection_result_from_file(filename):
    with open(filename,'rb') as f:
        cr = pickle.load(f)
    return cr


def CalculateAveragePrecision(rec, prec):
    #Copied from: https://github.com/rafaelpadilla/Object-Detection-Metrics
    #difference between this and the sklearn version is that it treats
    #precision-recall as non-increasing?
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

def calc_map(auth,splc):
#    y_true = np.concatenate((np.zeros(len(aG)),np.ones(len(sG))))
#    y_score = np.concatenate((aG,sG))
#
#    ap1 = average_precision_score(y_true, y_score)
#    ap2 = average_precision_score(1-y_true, -1*y_score)
    y_true = np.concatenate((np.zeros(len(auth)),np.ones(len(splc))))
    y_score = np.concatenate((auth,splc))
    
    #average precision for spliced images
    pp,rr,_ = precision_recall_curve(y_true,y_score)
    ap1 = CalculateAveragePrecision(np.flip(rr),np.flip(pp))
    
    #average precision for authentic images
    pp,rr,_ = precision_recall_curve(1-y_true,-1*y_score)
    ap2 = CalculateAveragePrecision(np.flip(rr),np.flip(pp))
    
    #mean average precision
    meanAP = np.mean((ap1[0],ap2[0]))
    return meanAP

def roc(v0,v1,T=None):
    
    if T is None:
        #thresholds (calc at each auth and spliced metric point)
        T = np.sort(np.concatenate((v0,v1))) 
    
    pfa = []
    pd = []
    for t in T:
        fa = sum(v0>= t)/float(len(v0)) #false alarms at t
        pfa.append(fa)
        d = sum(v1 >= t)/float(len(v1)) #detections at t
        pd.append(d)
    vpfa = np.array(pfa)
    vpd = np.array(pd)
    
    return vpfa,vpd
