import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8'
import argparse
from data import *
from importlib import import_module
from GCN import GCN_predicter
from LGCL import LGCL
from BARA import BARA
import torch
import numpy as np
import time
from tensorboardX import SummaryWriter
from train import train,init_network 
import json

import datetime
seed =234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)   
np.random.seed(seed)  # Numpy module.
torch.manual_seed(seed)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--begin_year', type=int, default=2010, help='Start year for the dataset.')
    parser.add_argument('--end_year', type=int, default=2023, help='End year for the dataset.')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--dim', type=int, default=384, help='Dimensionality of features.')
    parser.add_argument('--dataset', type=str, default='dataL3', choices=['G', 'random', 'dataL3'], help='Dataset to use.')    
    parser.add_argument('--model_name', type=str, default='LGCL', choices=['LGCL', 'GCN'], help='Model to use.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--conv_hidden', type=int, default=256, help='Hidden size for convolution layers.')
    parser.add_argument('--conv_out', type=int, default=32, help='Output size for convolution layers.')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of heads for the transformer.')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers for the transformer.')
    parser.add_argument('--transformer_ffn_hidden', type=int, default=64, help='Hidden size for transformer FFN.')
    parser.add_argument('--transformer_out', type=int, default=32, help='Output size for transformer.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--eta', type=float, default=0.7, help='Hyperpremeter Eta.')
    parser.add_argument('--no_co',action='store_true', help='Flag to disable co-occurrence view.')
    parser.add_argument('--wo_LLM', action='store_true', help='Flag to disable large language model usage.')
    parser.add_argument('--wo_DL', action='store_true', help='Flag to disable distangled loss.')
    parser.add_argument('--wo_KFP', action='store_true', help='Flag to use simple mm.')

    parser.add_argument('--state_path', type=str, default=None, help='Path to save the state of the model.')
    parser.add_argument('--log_path', type=str, default=None, help='Path to save the logs.')

    args = parser.parse_args()

    return args

class Config(object):
    '''
    配置参数
    '''
    def __init__(self, args):
        self.begin = args.begin_year
        self.end = args.end_year
        self.num_epochs = args.num_epochs
        self.dim = args.dim
        self.dataset = args.dataset
        self.model_name = args.model_name

        if self.dataset == 'G':
            self.num = 38847
        elif self.dataset == 'random':
            self.num = 20000
        elif self.dataset == 'dataL3':
            self.num = 678
        self.learning_rate = args.learning_rate
        self.conv_hidden = args.conv_hidden
        self.conv_out = args.conv_out
        self.num_heads = args.num_heads
        self.num_encoder_layers = args.num_encoder_layers
        self.transformer_ffn_hidden = args.transformer_ffn_hidden
        self.transformer_out = args.transformer_out
        self.dropout = args.dropout
        self.device = args.device
        self.lstm_hidden = 128
        self.eta = args.eta
        self.wo_LLM = args.wo_LLM
        self.no_co = args.no_co
        self.wo_DL = args.wo_DL
        self.wo_KFP = args.wo_KFP

        self.state_path = args.state_path
        self.LOG = args.log_path 


if __name__ == '__main__':
    args = get_args()
    config = Config(args)
    begin_all  = config.begin 
    end_all = config.end
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(device.type)
    model_name = config.model_name
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  

    print('config:')
    print('dataset:', config.dataset)
    print('begin:',config.begin)
    print('end:',config.end)    
    print('learning_rate:',config.learning_rate)
    print('dim:',config.dim)
    print('model_name:',config.model_name)
    print('state_path:',config.state_path)


    if config.dataset == 'dataL3':

        starttime = datetime.datetime.now()
        R = get_R_L3(begin=begin_all,end =end_all)   ##relation matrix
        endtime = datetime.datetime.now()
        print("R finished!")
        print (endtime - starttime)

        starttime = datetime.datetime.now()
        if config.model_name == 'LGCL':
            if config.wo_LLM == False:
                feature_flow = get_feature_L3(name = 'flow')
                feature_co = get_feature_L3(name = 'co')
                feature = [feature_flow,feature_co]
                feature_o = compare_get_feature_L3()
                corrected_flow = BARA(
                                    primary = feature_flow,
                                    fallback = feature_o,
                                    device  = device,
                                    epochs  = 80,
                                    lr      = 1e-3
                                )
                corrected_co = BARA(
                                    primary = feature_co,
                                    fallback = feature_o,
                                    device  = device,
                                    epochs  = 80,
                                    lr      = 1e-3
                                )
                 feature = [corrected_flow, corrected_co]
                
            else:

                feature = compare_get_feature_L3()
                feature = [feature,feature]
            
        else: 
            feature = compare_get_feature_L3(name = 'flow')
        endtime = datetime.datetime.now()
        print("feature finished!")
        print (endtime - starttime)

        starttime = datetime.datetime.now()
        co = get_co_L3(begin=begin_all,end =end_all)
        endtime = datetime.datetime.now()
        print("co_class finished!")
        print (endtime - starttime)

    if config.dataset == 'G':

        starttime = datetime.datetime.now()
        R = get_R_G(begin=begin_all,end =end_all)   ##relation matrix
        endtime = datetime.datetime.now()
        print("R finished!")
        print (endtime - starttime)

        starttime = datetime.datetime.now()
        if config.model_name == 'LGCL':
            if config.wo_LLM == False:
                feature_flow = get_feature_g(name = 'flow')
                feature_co = get_feature_g(name = 'co')
                feature = [feature_flow,feature_co]
                feature_o = compare_get_feature_G()
                corrected_flow = BARA(
                                    primary = feature_flow,
                                    fallback = feature_o,
                                    device  = device,
                                    epochs  = 80,
                                    lr      = 1e-3
                                )
                corrected_co = BARA(
                                    primary = feature_co,
                                    fallback = feature_o,
                                    device  = device,
                                    epochs  = 80,
                                    lr      = 1e-3
                                )
                feature = [corrected_flow, corrected_co] 
            else:
                feature = compare_get_feature_G()
                feature = [feature,feature]
        else: 
            feature = compare_get_feature_G(name = 'flow')
        endtime = datetime.datetime.now()
        print("feature finished!")
        print (endtime - starttime)

        starttime = datetime.datetime.now()
        co = get_co_G(begin=begin_all,end =end_all)
        endtime = datetime.datetime.now()
        print("co_class finished!")
        print (endtime - starttime)

    if config.dataset == 'random':

        starttime = datetime.datetime.now()
        R = get_R_random(begin=begin_all,end =end_all)   ##relation matrix
        endtime = datetime.datetime.now()
        print("R finished!")
        print (endtime - starttime)

        starttime = datetime.datetime.now()
        if config.model_name == 'LGCL':
            if config.wo_LLM == False:
                feature_flow = get_feature_random(name = 'flow')
                feature_co = get_feature_random(name = 'co')
                feature = [feature_flow,feature_co]
                feature_o = compare_get_feature_random()
                corrected_flow = BARA(
                                    primary = feature_flow,
                                    fallback = feature_o,
                                    device  = device,
                                    epochs  = 80,
                                    lr      = 1e-3
                                )
                corrected_co = BARA(
                                    primary = feature_co,
                                    fallback = feature_o,
                                    device  = device,
                                    epochs  = 80,
                                    lr      = 1e-3
                                )
                feature = [corrected_flow, corrected_co] 
            else:
                feature = compare_get_feature_random()
                feature = [feature,feature]
        else: 
            feature = compare_get_feature_random(name = 'flow')
        endtime = datetime.datetime.now()
        print("feature finished!")
        print (endtime - starttime)

        starttime = datetime.datetime.now()
        co = get_co_random(begin=begin_all,end =end_all)
        endtime = datetime.datetime.now()
        print("co_class finished!")
        print (endtime - starttime)


    online_model = LGCL( device,config ).to(device)
    target_model = LGCL( device,config ).to(device)

    writer = SummaryWriter(config.LOG)

    train(online_model,target_model, config,feature ,R, co, writer, device)

