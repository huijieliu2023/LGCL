
import copy
import numpy as np
import networkx as nx
import torch
import pandas as pd
import itertools
import csv
import time 
from datetime import timedelta
import scipy
import scipy.sparse as sparse
from scipy.sparse import csr_matrix,coo_matrix

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range



def get_link_labels(pos_edge_index, neg_edge_index,device):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def pos_link_labels(pos_edge_index,device):
    E = pos_edge_index.size(1) 
    link_labels = torch.ones(E, dtype=torch.float, device=device)
    return link_labels

def neg_link_labels(neg_edge_index,device):
    E = neg_edge_index.size(1) 
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    return link_labels


def refind_subclass(number,dict_resubclass):
    if number in dict_resubclass.keys():
        return dict_resubclass[number]
    else:
        return "none"
    

def calculate_mrr(y_pred_pos_list, y_pred_neg1_list, y_pred_neg2_list):

    assert len(y_pred_pos_list) == len(y_pred_neg1_list) == len(y_pred_neg2_list)
    
    reciprocal_ranks = []
    hits_at_1 = []
    hits_at_2 = []
    for y_pred_pos, y_pred_neg1, y_pred_neg2 in zip(y_pred_pos_list, y_pred_neg1_list, y_pred_neg2_list):
        scores = [y_pred_pos, y_pred_neg1, y_pred_neg2]
        scores_values = [score.item() for score in scores]
        sorted_indices = np.argsort(scores_values)[::-1]  
        rank = list(sorted_indices).index(0) + 1  
        reciprocal_rank = 1 / rank
        reciprocal_ranks.append(reciprocal_rank)
        if rank <= 1:
            hits_at_1.append(1)
        else:
            hits_at_1.append(0)
        if rank <= 2:
            hits_at_2.append(1)
        else:
            hits_at_2.append(0)
    mrr_value = sum(reciprocal_ranks) / len(reciprocal_ranks)
    hit_1_value = sum(hits_at_1) / len(hits_at_1)
    hit_2_value = sum(hits_at_2) / len(hits_at_2)
    return mrr_value,hit_1_value, hit_2_value



def compute_rate(R_train,device,data_class):

    cpc_title = pd.read_csv('g_cpc_title 2.tsv',sep='\t')
    if data_class == 'dataL3':
        subclass=list(set(cpc_title['cpc_subclass']))
        
    else:
        subclass=list(set(cpc_title['cpc_group'].dropna()))
    subclass.sort()
    dict_subclass = {}
    for i in range(len(subclass)):
        dict2 = {subclass[i]: i }
        dict_subclass.update(dict2)

    dict_resubclass = {}
    for i in range(len(subclass)):
        dict3 = {i:subclass[i]}
        dict_resubclass.update(dict3)    


    ref_posedge1 = torch.tensor(R_train[-2].nonzero(),dtype=torch.long).to(device)
    ref_posedge= ref_posedge1
    ref_heterophilic_edge = []
    ref_homophilic_edge = []                 
    output_file = "random_numbers.npy"
    loaded_numbers = np.load(output_file)
    loaded_numbers.sort()                                                                         
    for i in range(len(ref_posedge[0])):
        if data_class == 'dataL3':
            name_s = refind_subclass(int(ref_posedge[0][i]),dict_resubclass)
            name_d = refind_subclass(int(ref_posedge[1][i]),dict_resubclass)
        else:
            name_s = refind_subclass(loaded_numbers[int(ref_posedge[0][i])],dict_resubclass)
            name_d = refind_subclass(loaded_numbers[int(ref_posedge[1][i])],dict_resubclass)
            
        if name_s[0]!= name_d[0]:
            ref_heterophilic_edge.append(i)
        else: 
            ref_homophilic_edge.append(i)
    ref_hete_rate = len(ref_heterophilic_edge)/(len(ref_heterophilic_edge)+len(ref_homophilic_edge))   
    ref_homo_rate = len(ref_homophilic_edge)/(len(ref_heterophilic_edge)+len(ref_homophilic_edge)) 
    return ref_hete_rate, ref_homo_rate


def get_hetero_homo_edge(posedge,data_class):
    cpc_title = pd.read_csv('g_cpc_title 2.tsv',sep='\t')
    if data_class == 'dataL3':
        subclass=list(set(cpc_title['cpc_subclass']))
        subclass.sort()
    else:
        subclass=list(set(cpc_title['cpc_group']))
    dict_subclass = {}
    for i in range(len(subclass)):
        dict2 = {subclass[i]: i }
        dict_subclass.update(dict2)

    dict_resubclass = {}
    for i in range(len(subclass)):
        dict3 = {i:subclass[i]}
        dict_resubclass.update(dict3)    

    heterophilic_edge = []
    homophilic_edge = []                                                                                          
    for i in range(len(posedge[0])):
        if data_class == 'dataL3':
            name_s = refind_subclass(int(posedge[0][i]),dict_resubclass)
            name_d = refind_subclass(int(posedge[1][i]),dict_resubclass)
        else:
            output_file = "random_numbers.npy"
            loaded_numbers = np.load(output_file)
            loaded_numbers.sort()
            name_s = refind_subclass(loaded_numbers[int(posedge[0][i])],dict_resubclass)
            name_d = refind_subclass(loaded_numbers[int(posedge[1][i])],dict_resubclass)
        if name_s[0]!= name_d[0]:
            heterophilic_edge.append(i)
        else: 
            homophilic_edge.append(i)
    return heterophilic_edge, homophilic_edge


