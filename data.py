import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple
import json
import numpy as np
from scipy.sparse import csr_matrix,coo_matrix
import scipy.sparse as sp
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import networkx as nx
from deepsnap.graph import Graph
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv 
from torch_geometric.utils import train_test_split_edges
from tqdm import tqdm
import pandas as pd
from scipy.sparse import diags
from utils import refind_subclass
import csv
import random
seed=2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import scipy
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from networkx import convert_matrix  
from sklearn import preprocessing


def npz_threshold(npz):
    threshold =int(np.percentile(npz.data, (80), method='midpoint'))
    dat = npz.data[npz.data>=threshold]
    row = npz.row[npz.data>=threshold]
    col = npz.col[npz.data>=threshold]
    new_npz =  coo_matrix(((dat),(row,col)), shape = (npz.shape[0],npz.shape[0]))
    return new_npz
    


def get_G_indices(matrix):
    g_indices = list(range(171232, 210079))
    group_matrix_csr = matrix.tocsr()
    sub_matrix_g = group_matrix_csr[g_indices, :][:, g_indices]
    return sub_matrix_g.tocoo()


def get_random_indices(matrix):
    output_file = "random_numbers.npy"
    loaded_numbers = np.load(output_file)
    loaded_numbers.sort()
    group_matrix_csr = matrix.tocsr()
    sub_matrix_gh = group_matrix_csr[loaded_numbers, :][:, loaded_numbers]
    return sub_matrix_gh.tocoo()



def Normalize(array):
    mx = np.nanmax(array)
    mn = np.nanmin(array)
    t = (array-mn)/(mx-mn)
    return t




def get_R_L3(begin=2000,end=2007):
    R=[]
    for i in range(begin,end+1):
        R.append(scipy.sparse.load_npz('CPC-L3/TKF/'+str(i)+'npz'))
    return R

def get_co_L3(begin=2000, end= 2007):
    co=[]
    for i in range(begin,end+1):
        co.append(scipy.sparse.load_npz('CPC-L3/CO/'+str(i)+'npz'))
    return co


def get_R_G(begin=2000,end=2007):
    R=[]
    for i in range(begin,end+1):
        matrix = scipy.sparse.load_npz('CPC-G/TKF/'+str(i)+'npz')
        g_indices = list(range(171232, 210079)) 
        group_matrix_csr = matrix.tocsr()
        sub_matrix_g = group_matrix_csr[g_indices, :][:, g_indices]
        sub_matrix_g = sub_matrix_g.tocoo()
        R.append(sub_matrix_g)
    return R


def get_co_G(begin=2000, end= 2007):
    co=[]
    for i in range(begin,end+1):
        matrix = scipy.sparse.load_npz('CPC-G/CO/'+str(i)+'npz')
        g_indices = list(range(171232, 210079)) 
        group_matrix_csr = matrix.tocsr()
        sub_matrix_g = group_matrix_csr[g_indices, :][:, g_indices]
        sub_matrix_g = sub_matrix_g.tocoo()
        co.append(sub_matrix_g)
    return co



def get_R_random(begin=2000,end=2007):
    R=[]
    for i in range(begin,end+1):
        matrix = scipy.sparse.load_npz('CPC-R/TKF/'+str(i)+'npz')
        sub_matrix_gh = get_random_indices(matrix)
        R.append(sub_matrix_gh)
    return R

def get_co_random(begin=2000, end= 2007):
    co=[]
    for i in range(begin,end+1):
        matrix = scipy.sparse.load_npz('CPC-R/CO/'+str(i)+'npz')
        sub_matrix_gh = get_random_indices(matrix)
        co.append(sub_matrix_gh)
    return co

def get_feature_L3(name='flow'):
    with open('CPC-L3/Text/emb_qw_'+str(name)+'.json','r', encoding='utf-8') as fp:
        subclass_emb = []
        for line in fp:
            dic = json.loads(line)
            subclass_emb.append(dic)  
    subclass_emb = np.array(subclass_emb)
    subclass_emb = torch.from_numpy(subclass_emb).float()

    return subclass_emb

def get_feature_random(name='flow'):
    with open("CPC-R/Text/emb_qw_random_"+str(name)+".json",'r', encoding='utf-8') as fp:
        emb = []
        for line in fp:
            dic = json.loads(line)
            emb.append(dic)  
    emb = np.array(emb)
    emb = torch.from_numpy(emb).float()
    return emb

def get_feature_g(name='flow'):
    with open("CPC-G/Text/emb_qw_g_"+str(name)+".json",'r', encoding='utf-8') as fp:
        emb = []
        for line in fp:
            dic = json.loads(line)
            emb.append(dic)  
    emb = np.array(emb)
    emb = torch.from_numpy(emb).float()
    return emb

def compare_get_feature_L3():
    with open("CPC-L3/Text/emb_subclass_original.json",'r', encoding='utf-8') as fp:
        subclass_emb = []
        for line in fp:
            dic = json.loads(line)
            subclass_emb.append(dic)  
    subclass_emb = np.array(subclass_emb)
    subclass_emb = torch.from_numpy(subclass_emb).float()
    return subclass_emb


def compare_get_feature_G():

    with open("CPC-G/Text/emb_L4_original_g.json",'r', encoding='utf-8') as fp:
        emb = []
        for line in fp:
            dic = json.loads(line)
            emb.append(dic)  
    emb = np.array(emb)
    emb = torch.from_numpy(emb).float()
    return emb

def compare_get_feature_random():

    with open("CPC-R/Text/emb_L4_original_random.json",'r', encoding='utf-8') as fp:
        emb = []
        for line in fp:
            dic = json.loads(line)
            emb.append(dic)  
    emb = np.array(emb)
    emb = torch.from_numpy(emb).float()
    return emb



