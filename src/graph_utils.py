#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:53:09 2019

@author: owen
"""


import numpy as np
import igraph

def sym_mat(mat,diag_val = None):
    mat = 0.5*(mat + mat.transpose())
    if not (diag_val is None):
        np.fill_diagonal(mat,diag_val)
    return mat

def adj_to_graph(mat,threshold=0):
    g = igraph.Graph.Adjacency((mat>threshold).tolist())
    g.to_undirected()
    g.es['weight'] = mat[mat>threshold]
    g.vs["label"] = range(len(mat))
    return g

def cluster_fastgreedy(g,weighted=True,n=None):
    
    if weighted is True:
        com = g.community_fastgreedy(weights=g.es['weight']).as_clustering(n=n)
    else:
        com = g.community_fastgreedy().as_clustering(n=n)
        
    n_clusters = len(np.unique(com.membership))
    modularity  = com.modularity
    return com, n_clusters, modularity


def glist_to_fastgreedy_modularity(glist,threshold=0.0,weighted=True):
    list_mod = []
    for m in glist:
        normed_mat = sym_mat(m[1],diag_val=1) #make sure it's symmetric
        g = adj_to_graph(normed_mat,threshold=threshold)
        _,_,modularity = cluster_fastgreedy(g,weighted=weighted)
        list_mod.append(modularity)
    return list_mod
