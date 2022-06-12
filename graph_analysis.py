# -*- coding: utf-8 -*-

"""
Graph clustering and graph classification
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import scipy as sp
import numpy as np
import networkx as nx
import torch_geometric
from torch_geometric.datasets import TUDataset
from scipy.sparse import csr_matrix
from typing import Dict, Tuple

from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn import svm

from methods.gromov_funcs import define_loss_function, gw_distance, spar_gw

np.random.seed(123)


#############################################################################
#
# Define the loss function
# ---------------------------------------------

loss_func_name = 'square_loss'
loss_func = define_loss_function(loss_func_name)


if loss_func_name == 'square_loss':
    gromov_loss_func = loss_func_name
else:
    gromov_loss_func = loss_func
    

#############################################################################
#
# Load the dataset
# ---------------------------------------------

dataset = TUDataset(root='/tmp/SYNTHETIC', name='SYNTHETIC', use_node_attr=True)
num_graph = len(dataset)
num_class = dataset.num_classes
num_feature = dataset.num_node_features
y_true = []
num_node = []
for data in dataset:
    y_true.append(data.y.numpy())
    num_node.append(data.num_nodes)
print('No. of nodes. min: ', min(num_node), 'max: ', max(num_node), 'ave: ', np.mean(num_node))


def extract_TUDataset_graph(data: torch_geometric.data.data) -> Tuple[np.ndarray, csr_matrix, np.ndarray, Dict]:
    """
    Extract the node distribution, adjacency matrix, feature matrix, and node index dictionary from the graph in TUDataset
    Args:
        graph: the graph instance generated via networkx
    Returns:
        probs: the distribution of nodes (n, )
        adj: adjacency matrix (n, n)
        x: nodes feauture matrix (n, num_feature)
        idx2node: a dictionary {key: idx, value: node name}
    """
    x, edge_index = data.x.numpy(), data.edge_index.numpy()
    G = nx.Graph()
    for ii in range(edge_index.shape[1]):
        G.add_edge(edge_index[0,ii], edge_index[1,ii])
    adj = nx.adjacency_matrix(G)
    degrees = np.array(list(G.degree))[:, 1]
    probs = degrees / np.sum(degrees)
    
    idx2node = {}
    for i in range(len(probs)):
        idx2node[i] = i
    
    return probs, csr_matrix(adj), x, idx2node


#############################################################################
#
# Compute the distance matrix
# ---------------------------------------------
epsilon = 1e-2
alpha = 0.6

dist_mat = np.zeros((num_graph, num_graph))

for i in range(0, num_graph-1):
    print('i: ', i)
    
    p_s, cost_s, x_s, idx2node_s = extract_TUDataset_graph(dataset[i])
    C1 = cost_s.toarray().astype(float)

    for j in range(i+1, num_graph):
        
        p_t, cost_t, x_t, idx2node_t = extract_TUDataset_graph(dataset[j]) 

        M = sp.spatial.distance.cdist(x_s, x_t)
        M /= np.max(M)

        trans = spar_gw(cost_s, cost_t, p_s, p_t, gromov_loss_func, 2**5*max(M.shape), 
                        epsilon, M, alpha, random_state=123+100*j+421*i)
        
        dist_mat[i,j] = gw_distance(cost_s, cost_t, gromov_loss_func, trans, M, alpha)


#############################################################################
#
# Graph clustering
# ---------------------------------------------

num_trial = 10
ri_list = np.zeros((len(range(-10,11)), num_trial))
ari_list = np.zeros((len(range(-10,11)), num_trial))

for ii in range(-10,11):
    gamma = 2**ii
    kernel_mat = np.exp(-dist_mat/gamma)
    
    for jj in range(num_trial):
        sc = SpectralClustering(n_clusters=num_class, 
                                random_state=1024+4321*jj,
                                affinity='precomputed',
                                assign_labels='discretize')
        y_pred = sc.fit_predict(kernel_mat)
    
        rand_index = metrics.rand_score(y_true, y_pred)
        ri_list[ii,jj] = rand_index


mean_ri_list = np.nanmean(ri_list, axis=1)
ri_max = np.max(mean_ri_list)
ind_max = np.argmax(mean_ri_list)
ri_std = np.std(ri_list[ind_max,])
print('(ri) mean:', ri_max*100, 'std:', ri_std*100)


#############################################################################
#
# Graph classification
# ---------------------------------------------

shuffle = np.random.choice(range(num_graph), size=num_graph, replace=False)
n_fold = 5
size_fold = int(num_graph/n_fold)
acc_list = np.zeros((len(range(-10,11)), n_fold))

for ii in range(-10,11):
    gamma = 2**ii
    kernel_mat = np.exp(-dist_mat/gamma)
    
    for jj in range(n_fold):
        test_ind = shuffle[(jj*size_fold):((jj+1)*size_fold)]
        train_ind = np.delete(shuffle, range((jj*size_fold), ((jj+1)*size_fold)))
        
        kernel_mat_train = kernel_mat[train_ind,][:,train_ind]
        kernel_mat_test = kernel_mat[test_ind,][:,train_ind]
        y_train = y_true[train_ind]
        y_test = y_true[test_ind]

        clf = svm.SVC(kernel="precomputed")
        clf.fit(kernel_mat_train, y_train)
        y_pred = clf.predict(kernel_mat_test)
        
        acc_list[ii,jj] = np.sum(y_test==y_pred)/size_fold
        
        
mean_acc_list = np.nanmean(acc_list, axis=1)
acc_max = np.max(mean_acc_list)
ind_max = np.argmax(mean_acc_list)
acc_std = np.std(acc_list[ind_max,])
print('mean:', acc_max*100, 'std:', acc_std*100)
