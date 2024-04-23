#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: owen
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

import tensorflow as tf
from .blockimage import tile_image, span_image_by_overlap
import seaborn as sns

import pickle

class SimGraph:
    """Similarity Graph Class

    forensic similarity graph for image forgery detection and localization

    Attributes
    ----------
    mat : NxN numpy array, float
        matrix containing pairwise forensic similarity between image patches
    inds : list of (x,y) int tuples
        list of x,y coordinates that correspond to the top-left corner of each 
        inspected patch in the image. len(inds) = N
    patch_size : int
        size of the patch used

    Parameters
    ----------
    mat : NxN numpy array, float
        matrix containing pairwise forensic similarity between image patches
    inds : list of (x,y) int tuples
        list of x,y coordinates that correspond to the top-left corner of each 
        inspected patch in the image. len(inds) = N
    patch_size : int
        size of the patch used
    """
    
    def __init__(self,mat,inds,patch_size):
        self.mat = mat
        self.inds = inds
        self.patch_size = patch_size
        
    #def to_igraph(weighted=True,threshold=0.0): #convert to igraph representation
    #def to_file() #save to pickle
    
    def plot_matrix(self,palette='GnBu'):
        fig,ax = plt.subplots(1)

        sns.heatmap(self.mat,ax=ax,cmap=sns.color_palette(palette=palette,n_colors=100),
            		    cbar_kws={'label':'Similarity'},linewidths=0.0,vmin=0,vmax=1)
            
        plt.yticks(rotation=0);
        plt.ylabel('Patch 1 Index')
        plt.xlabel('Patch 2 Index')
        plt.tight_layout()
        return fig, ax
    
    def plot_indices(self,ax=None,image=None,fontsize=10):
        if (ax is None) and (image is None): #if don't get an axis to plot on, create one
            fig,ax = plt.subplots(1)
        
        #if we get an image to plot on, create figure with image in it
        if image is not None:
            fig,ax = plt.subplots(1) #create figure and axis
            ax.imshow(image) #plot image
            plt.xticks([]) #remove ticks
            plt.yticks([])
            plt.tight_layout() #remove whitespace

        for i,ind in enumerate(self.inds): #iterate through each index
            #write index on center of its location
    	    tt = ax.text(ind[0]+self.patch_size/2, ind[1]+self.patch_size/2,str(i),
                      color='white', fontsize=fontsize)
            #outline in black so its easy to read
    	    tt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                              path_effects.Normal()])
        return ax
    
    def save(self,name):
        with open(name,'wb') as f:
            pickle.dump(self,f,pickle.HIGHEST_PROTOCOL)
                
                               
def from_file(filename):
    with open(filename,'rb') as f:
        sg = pickle.load(f)
    return sg

def softmax(a): #function to calculate softmax
    e = np.exp(a)
    div = np.tile(e.sum(1,keepdims=1),(1,a.shape[1]))
    sm = e/div
    return sm
    
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
    

def calc_simgraph_insession(sess, compare_output, x1, x2, MISL_phase, image, patch_size, overlap, span=True, batch_size = 48, quiet = False):

	if isinstance(image,str): #if input is a string, assume it's a file path
		image = plt.imread(image) #load image
    
	if isinstance(overlap,tuple):
		dx = overlap[0]
		dy = overlap[1]
	else:
		dx = overlap
		dy = overlap
        
        
    #tile image
	if span:
		X, inds = span_image_by_overlap(image,patch_size, patch_size, dx, dy, snap_to=16)
	else:
		X,inds = tile_image(image, patch_size, patch_size, dx, dy)

	xx2 = X #patch 2 is all of the patches in the image
	listsm_x1 = [] #intialize list (these will be rows of the matrix)
	for ind in tqdm(range(len(X)),disable=quiet): #iterate over each patch

		#create a vector of patch 1, same length as xx2
		xx1 = np.tile(X[ind],(xx2.shape[0],1,1,1)) 
		listsm_x2 = [] #initialize list, once populated this will be a single row

		for ix in batch(range(len(xx1)), batch_size): #batchify
			#calculate output
			result = sess.run(compare_output, feed_dict={x1:xx1[ix],x2:xx2[ix], MISL_phase:False})
			#make into a list			
			listsm_x2.append(softmax(result))

		#make into a numpy array
		smx2 = np.concatenate(listsm_x2)
		#add to rows, take similarity value only
		listsm_x1.append(smx2[:,1])
	#numpyify
	mat = np.vstack(listsm_x1)
	sg = SimGraph(mat,inds,patch_size)
	return sg
    #return mat, inds

#TWO-STEP Process
def calc_features_insession(X,sess,mislnet_features,x,MISL_phase,batch_size=48):
    list_feats = []
    for ix in batch(range(len(X)),batch_size):
        result = sess.run(mislnet_features, feed_dict={x:X[ix], MISL_phase:False})
        list_feats.append(result)
    feats = np.concatenate(list_feats)
    return feats

def compare_features_insession(F1,F2,sess,compare_output,f1,f2,batch_size=10,quiet=False):
    listsm = []
    xx2 = F2
    for ind in tqdm(range(len(F1)),disable=quiet):
        xx1 = np.tile(F1[ind],(xx2.shape[0],1))
        for ix in batch(range(len(xx1)),batch_size):
            result = sess.run(compare_output, feed_dict={f1:xx1[ix],f2:xx2[ix]})
            listsm.append(softmax(result))
    sm = np.concatenate(listsm)
    return sm

def calc_simgraph_insession_twostep(sess, mislnet_feats, compare_output, x, f1, f2, MISL_phase,
                                    image, patch_size, overlap, span=True, image_batch = 48,feature_batch = 2056, quiet = False):
    
    if isinstance(image,str): #if input is a string, assume it's a file path
        image = plt.imread(image) #load image
    	#tile image
      
    #overlap logic, overlap can either be a tuple=(dx,dy), or an int dx=dy=overlap
    if isinstance(overlap,tuple):
        dx = overlap[0]
        dy = overlap[1]
    else:
        dx = overlap
        dy = overlap
    
    #tile image
    if span:
        X, inds = span_image_by_overlap(image,patch_size, patch_size, dx, dy, snap_to=16)
    else:
        X,inds = tile_image(image, patch_size, patch_size, dx, dy)
    
    feats = calc_features_insession(X,sess,mislnet_feats,x,MISL_phase,batch_size=image_batch)
    sm = compare_features_insession(feats,feats,sess,compare_output,f1,f2,batch_size=feature_batch,quiet=quiet)
    
    sim = np.array(sm)[:,1] #similarity neuron
    mat = sim.reshape((len(X),len(X))) #convert to matrix form
    sg = SimGraph(mat,inds,patch_size)
    return sg

def calc_simgraph(image,f_weights_restore,patch_size,overlap,run_type='twostep',
                  span=True,batch_size=48,feature_batch=2056,quiet=False):
    
    if isinstance(image,str): #if input is a string, assume it's a file path
        image = plt.imread(image) #load image
    
    if run_type == 'twostep':
        #Load CompareNet Model
        if patch_size == 256:
        	from .mislnet_model import MISLNet 
        elif patch_size == 128:
        	from .mislnet_model import MISLNet128 as MISLNet
        elif patch_size == 96:
        	from .mislnet_model import MISLNet96 as MISLNet
        elif patch_size == 64:
        	from .mislnet_model import MISLNet64 as MISLNet
        else:
            raise TypeError('Unsupported patch size {}'.format(patch_size))
        
        from .mislnet_model import prefeat_CompareNet_v1
        
        #reset tf
        tf.reset_default_graph()    
        #PLACE HOLDERS
        x = tf.placeholder(tf.float32, shape=[None,patch_size,patch_size,3], name='input_data')
        f1 = tf.placeholder(tf.float32, shape=[None,200], name='feature1')
        f2 = tf.placeholder(tf.float32, shape=[None,200], name='feature2')
        MISL_phase =tf.placeholder(tf.bool, name='phase')
        
        mislnet_feats = MISLNet(x,MISL_phase,nprefilt=6)
        mislnet_compare = prefeat_CompareNet_v1(f1,f2)
        
        mislnet_restore = tf.train.Saver()
        
        with tf.Session() as sess:
            mislnet_restore.restore(sess,f_weights_restore) #load pretrained network
            sg = calc_simgraph_insession_twostep(sess, mislnet_feats, mislnet_compare, x, f1, f2, MISL_phase,
                                                       image, patch_size, overlap, span=span, image_batch = batch_size,feature_batch = feature_batch, quiet = quiet)
            sess.close()
        return sg
    
    elif run_type == 'onestep':
        #Load CompareNet Model
        if patch_size == 256:
        	from .mislnet_model import CompareNet_v1_256 as CompareNet
        elif patch_size == 128:
        	from .mislnet_model import CompareNet_v1_128 as CompareNet
        elif patch_size == 96:
        	from .mislnet_model import CompareNet_v1_96 as CompareNet
        elif patch_size == 64:
        	from .mislnet_model import CompareNet_v1_64 as CompareNet
        else:
            raise TypeError('Unsupported patch size {}'.format(patch_size))
        
        #reset tf
        tf.reset_default_graph()    
        #PLACE HOLDERS
        x1 = tf.placeholder(tf.float32, shape=[None,patch_size,patch_size,3], name='input_data1')
        x2 = tf.placeholder(tf.float32, shape=[None,patch_size,patch_size,3], name='input_data2')
        MISL_phase =tf.placeholder(tf.bool, name='phase')
        #CREATE NETWORK
        compare_output = CompareNet(x1,x2,MISL_phase)
        #initialize saver
        mislnet_restore = tf.train.Saver()
        
        with tf.Session() as sess:
            mislnet_restore.restore(sess,f_weights_restore) #load pretrained network
            sg = calc_simgraph_insession(sess, compare_output, x1, x2, MISL_phase, image, patch_size, overlap, span=span, batch_size = batch_size, quiet = quiet)
            sess.close()
        
        
        return sg
