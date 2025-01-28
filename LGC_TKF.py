import scipy
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import time
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv,GATConv,SAGEConv 
from utils import get_time_dif,get_link_labels,get_DAD,sparse_mx_to_torch_sparse_tensor

class LGC_TKF(torch.nn.Module):
    def __init__(self, device, config):
        super(LGC_TKF,self).__init__()

        self.relu = nn.LeakyReLU()
        self.device = device
        self.conv1 = GCNConv(config.dim, config.conv_hidden)
        self.conv2 = GCNConv(config.conv_hidden, config.conv_out)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.conv_out,  
            nhead=config.num_heads, 
            dim_feedforward=config.transformer_ffn_hidden,  
            dropout=config.dropout  
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers  
        )
        self.wo_KFP = config.wo_KFP
        self.lin = nn.Linear(config.conv_out, config.transformer_out)
        self.normalization = torch.nn.LayerNorm([config.num, config.dim])

        self.lstm = nn.LSTM(config.conv_out,
                            config.lstm_hidden,
                              2, #lstm_num_layers
                            batch_first=True)
        self.lin2 = nn.Linear(config.lstm_hidden, config.transformer_out )

        self.gru_hidden_size = 128
        self.gru = nn.GRU(config.conv_out,config.lstm_hidden,2)


        # Bilinear weight matrix for edge scoring
        self.bilinear_weight = nn.Parameter(torch.randn(config.transformer_out, config.transformer_out))

        # MLP for dynamic alpha weight calculation
        self.alpha_mlp = nn.Sequential(
            nn.Linear(2 * config.transformer_out, config.transformer_out),
            nn.ReLU(),
            nn.Linear(config.transformer_out, 1),  # Output a single value for alpha_ij
            nn.Sigmoid()  # Output in range (0, 1)
        )
#####
    def encode(self,E_0,edge):

        x = self.conv1(E_0, edge)
        x = x.relu()
        return self.conv2(x, edge)



    def trans_encode(self, x):
        ## x : torch.Size([11, 678, 32])
        out = self.transformer_encoder(x)
        out = self.lin(out)
        out = self.relu(out)
        return out



    def mm_simple(self,feature,pos_edge_index,neg_edge_index):
        edge_index = torch.cat([pos_edge_index,neg_edge_index],dim=-1)
        logits = (feature[edge_index[0]] * feature[edge_index[1]]).sum(dim=-1)
        logits = logits.to(self.device)

        return logits



    def mm(self, feature, pos_edge_index, neg_edge_index):
        # Concatenate positive and negative edge indices
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

        # Dynamic weight calculation using MLP
        h_i = feature[edge_index[0]]
        h_j = feature[edge_index[1]]
        concat_features = torch.cat([h_i, h_j], dim=-1)  # Concatenate node embeddings
        alpha_ij = self.alpha_mlp(concat_features)  # Get dynamic weight

        # Bilinear score
        bilinear_score = torch.einsum('nd,dd,nd->n', h_i, self.bilinear_weight, h_j)

        # Apply dynamic weight
        logits = bilinear_score * alpha_ij.squeeze(-1)
        logits = logits.to(self.device)

        return logits
    
    def forward(self,R,fea, pos_edge_index,neg_edge_index ):
        feature=[]
        fea = fea.to(self.device)
        fea = self.normalization(fea)
        for i in range(len(R)-1):
            edge = torch.tensor(R[i].nonzero(),dtype=torch.long).to(self.device)
            E_d0 = fea.to(self.device)
            E_end = self.encode(E_d0,edge )
            feature.append(E_end)
        feature = torch.stack(feature, dim = 0)
        output_d_f = self.trans_encode(feature)
        feature_pre = output_d_f[ -1,:, :]
        if self.wo_KFP == False:
            link_logits = self.mm(feature_pre,pos_edge_index,neg_edge_index)
        else:
            link_logits = self.mm_simple(feature_pre,pos_edge_index,neg_edge_index)
        return link_logits, feature




def contrastive_loss( z1, z2, temperature=0.3):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    N = z1.size(0)
    loss = -2 * torch.sum(torch.bmm(z1.view(N, 1, -1), z2.view(N, -1, 1)).squeeze()) / N
    return loss





def update_ema_parameters(online_model, target_model, tau=0.9):
    for online_param, target_param in zip(online_model.parameters(), target_model.parameters()):
        target_param.data = tau * target_param.data + (1 - tau) * online_param.data
