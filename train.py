# coding: UTF-8

from sched import scheduler
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import json
from data import *
from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score,f1_score,precision_score,recall_scor
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
import networkx as nx
import scipy
from networkx import convert_matrix  
from LGCL import *
import warnings
from ranking_metrics import *
import datetime

warnings.filterwarnings("ignore")


def train(online_model, target_model, config, feature, R,co,writer,device):
    start_time=time.time()
    init_network(online_model)
    state_path = config.state_path
    target_model.load_state_dict(online_model.state_dict())
    optimizer = torch.optim.Adam(params = online_model.parameters(),lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor= 0.5, patience=20, min_lr=0.00001) 
    R_train = R[0:len(R)-2]
    R_valid = R[1:len(R)-1]
    R_test = R[2:len(R)]

    co_train = co[0:len(co)-2]
    # co_valid = co[1:len(co)-1]
    # co_test = co[2:len(co)]

    posedge = torch.tensor(R_train[-1].nonzero(),dtype=torch.long).to(device)
    neg_edge_index1 = structured_negative_sampling(edge_index=torch.hstack((posedge,posedge)), num_nodes=R_train[-1].shape[0])
    neg_edge_index = torch.vstack((neg_edge_index1[0],neg_edge_index1[2]))

    starttime = datetime.datetime.now()
    if config.dataset == 'G':
        ref_hete_rate= 0  
        ref_homo_rate= 1
        heterophilic_edge = None 
        homophilic_edge = posedge
    else:
        ref_hete_rate, ref_homo_rate = compute_rate(R_train,device,data_class=config.dataset)
        heterophilic_edge, homophilic_edge = get_hetero_homo_edge(posedge,data_class=config.dataset)
    endtime = datetime.datetime.now()
    print('rate_ok!')
    print(ref_hete_rate,ref_homo_rate)
    print (endtime - starttime)

    link_labels_train = get_link_labels(posedge, neg_edge_index,device)
    if config.dataset =='G':
        link_labels_train_homophilic = pos_link_labels(posedge,device)
        link_labels_train_heterophilic = None
        link_labels_train_neg = neg_link_labels(neg_edge_index, device)
    else:
        link_labels_train_homophilic = pos_link_labels(posedge.T[homophilic_edge].T,device)
        link_labels_train_heterophilic = pos_link_labels(posedge.T[heterophilic_edge].T,device)
        link_labels_train_neg = neg_link_labels(neg_edge_index, device)

    best_auc = float(0)
    
    for epoch in tqdm(range(config.num_epochs)):
        print('Epoch [{}/{}]'.format(epoch , config.num_epochs))

        online_model.train()

        link_logits, feature_online = online_model(R_train,feature[0], posedge,neg_edge_index)
        _,feature_target = target_model(co_train,feature[1], posedge,neg_edge_index)


        link_logits = torch.reshape(link_logits,(link_logits.size(0),-1))
        link_labels = torch.reshape(link_labels_train,(link_labels_train.size(0),-1))  
        
        if config.dataset == 'G':
            link_labels_homophilic = torch.reshape(link_labels_train_homophilic,(link_labels_train_homophilic.size(0),-1))     
            link_labels_neg = torch.reshape(link_labels_train_neg,(link_labels_train_neg.size(0),-1))    
            loss_homo = F.binary_cross_entropy_with_logits(link_logits[0:len(posedge[0])], link_labels_homophilic).to(device)
            loss_neg = F.binary_cross_entropy_with_logits(link_logits[len(posedge[0]):], link_labels_neg).to(device)
            loss1 = F.binary_cross_entropy_with_logits(link_logits, link_labels).to(device)

            ####

            contrastive_loss_co_all = 0
            anchor_features = feature_online[-1].to(device)  
            co_features_last = feature_target[-1].to(device)
            contrastive_loss_co = contrastive_loss(anchor_features, co_features_last)
            contrastive_loss_co_all += contrastive_loss_co
            if config.no_co == False:
                if config.wo_DL == False:
                    loss = 2*ref_homo_rate * loss_homo+ loss_neg + config.eta*contrastive_loss_co_all
                else:
                    loss = loss1 + config.eta*contrastive_loss_co_all
                
            else:
                if config.wo_DL == False:
                    loss = 2*ref_homo_rate * loss_homo+ loss_neg 
                else:
                    loss = loss1

        
        else:
            link_labels_homophilic = torch.reshape(link_labels_train_homophilic,(link_labels_train_homophilic.size(0),-1))    
            link_labels_heterophilic = torch.reshape(link_labels_train_heterophilic,(link_labels_train_heterophilic.size(0),-1))    
            link_labels_neg = torch.reshape(link_labels_train_neg,(link_labels_train_neg.size(0),-1))    

            loss_homo = F.binary_cross_entropy_with_logits(link_logits[homophilic_edge], link_labels_homophilic).to(device)
            loss_hete = F.binary_cross_entropy_with_logits(link_logits[heterophilic_edge], link_labels_heterophilic).to(device)
            loss_neg = F.binary_cross_entropy_with_logits(link_logits[len(posedge[0]):], link_labels_neg).to(device)
            loss1 = F.binary_cross_entropy_with_logits(link_logits, link_labels).to(device)
            ####

            contrastive_loss_co_all = 0
            anchor_features = feature_online[-1].to(device) 
            co_features_last = feature_target[-1].to(device)
            contrastive_loss_co = contrastive_loss(anchor_features, co_features_last)
            contrastive_loss_co_all += contrastive_loss_co
            if config.no_co == False:
                if config.wo_DL == False:
                    loss = 2*ref_homo_rate * loss_homo+ 2*ref_hete_rate * loss_hete + loss_neg + config.eta*contrastive_loss_co_all
                else:
                    loss = loss1 + config.eta*contrastive_loss_co_all
            else:
                if config.wo_DL == False:
                    loss = 2*ref_homo_rate * loss_homo+ 2*ref_hete_rate * loss_hete + loss_neg
                else:
                    loss = loss1 


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        update_ema_parameters(online_model, target_model)

        link_labels = link_labels.cpu().detach().numpy()
        link_logits =  link_logits.cpu().detach().numpy()   
        train_auc = roc_auc_score(link_labels, link_logits)
        train_ap = average_precision_score(link_labels, link_logits)

        valid_loss , valid_auc ,valid_ap= valid(online_model ,feature[0], R_valid,device)
        if valid_auc  > best_auc:
            best_auc = valid_auc
            state = {'epoch': epoch, 'state_dict': online_model.state_dict(), 'best_auc':  best_auc,
                            'optimizer': optimizer.state_dict()}
            torch.save(state, state_path)

        time_dif = get_time_dif(start_time)
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.4},  Train Auc: {2:>6.2%}, Train AP: {3:>6.2%}, Valid Loss: {4:>5.4},  Valid Auc: {5:>6.2%}, Valid AP: {6:>6.2%},  Time: {7} '
        print(msg.format(epoch, loss.item(), train_auc,train_ap, valid_loss, valid_auc, valid_ap, time_dif))
        writer.add_scalar("loss/train", loss.item(), epoch)
        writer.add_scalar("loss/valid", valid_loss, epoch)
        writer.add_scalar("auc/train", train_auc, epoch)
        writer.add_scalar("auc/valid", valid_auc, epoch)
        writer.add_scalar("ap/train", train_ap, epoch)
        writer.add_scalar("ap/valid", valid_ap, epoch)

    writer.close()

    link_logits, link_labels,epoch,loss,pred_pos = test(online_model,feature[0], R_test, device,path=state_path)
    td(link_logits, link_labels,epoch,pred_pos,loss )


def valid(model,feature, R_valid,device):
    model.eval()

    posedge = torch.tensor(R_valid[-1].nonzero(),dtype=torch.long).to(device)
    neg_edge_index1 = structured_negative_sampling(edge_index=torch.hstack((posedge,posedge)), num_nodes=R_valid[-1].shape[0])
    neg_edge_index = torch.vstack((neg_edge_index1[0],neg_edge_index1[2]))

    link_logits,_ = model(R_valid,feature,posedge, neg_edge_index)
    link_labels = get_link_labels(posedge, neg_edge_index, device)
    link_logits = torch.reshape(link_logits,(link_logits.size(0),-1))
    link_labels = torch.reshape(link_labels,(link_labels.size(0),-1))     
    
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels).to(device)


    link_labels = link_labels.cpu().detach().numpy()
    link_logits =  link_logits.cpu().detach().numpy() 

    auc = roc_auc_score(link_labels,link_logits)
    ap = average_precision_score(link_labels, link_logits)

    return loss,auc,ap




def test(model,feature, R_test, device,path):
    model_state = torch.load(path)
    model.load_state_dict(model_state['state_dict'])
    epoch = model_state['epoch']
    model.eval()

    posedge = torch.tensor(R_test[-1].nonzero(),dtype=torch.long).to(device)
    neg_edge_index1 = structured_negative_sampling(edge_index=torch.hstack((posedge,posedge)), num_nodes=R_test[-1].shape[0])
    neg_edge_index = torch.vstack((neg_edge_index1[0],neg_edge_index1[2]))



    link_logits,_ = model(R_test,feature, posedge, neg_edge_index)

    link_labels = get_link_labels(posedge, neg_edge_index, device)
    link_logits = torch.reshape(link_logits,(link_logits.size(0),-1))
    link_labels = torch.reshape(link_labels,(link_labels.size(0),-1))       
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels).to(device)

    
    link_labels = link_labels.cpu().detach().numpy()
    link_logits =  link_logits.cpu().detach().numpy()   


    return link_logits, link_labels,epoch,loss,posedge


def td(link_logits, link_labels, epoch, pred_pos, loss ):

    #############
    y_pred_pos = link_logits[0:pred_pos.size()[1]]
    y_pred_neg1 = link_logits[pred_pos.size()[1]:pred_pos.size()[1]*2]
    y_pred_neg2 = link_logits[pred_pos.size()[1]*2:]

    mrr,hits1,hits2 = calculate_mrr(y_pred_pos,y_pred_neg1,y_pred_neg2 )

    auc = roc_auc_score(link_labels,link_logits)
    ap = average_precision_score(link_labels, link_logits)

    
    msg = 'Iter: {0:>6},  Test Loss: {1:>5.4},  Test Auc: {2:>6.2%}, Test AP: {3:>6.2%},\
            MRR:{4:>5.4}, HIT@1:{5:>6.2%},  HIT@2:{6:>6.2%} \n'
    print(msg.format(epoch, loss.item(), auc,ap, mrr, hits1, hits2))


