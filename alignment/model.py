import torch.nn as nn
import torch.nn.functional as F


import dgl

from .Module.GCN import GCN_encoder 
from .Module.GAT import GAT_encoder
from .Module.Graphormer import Graphormer_encoder

class alignment_model(nn.Module):
    def __init__(self, gnn_type, in_size, hid_size, out_size, heads, num_layers, dropout):
        super(alignment_model, self).__init__()
        
        self.gnn_type = gnn_type


        if gnn_type == 'Graphormer':
            self.encoder = Graphormer_encoder(in_size, hid_size, out_size, heads, num_layers, dropout)
        elif gnn_type == 'GAT':
            self.encoder = GAT_encoder(in_size, hid_size, out_size, heads, num_layers, dropout)
        elif gnn_type == 'GCN':
            self.encoder = GCN_encoder(in_size, hid_size, out_size, num_layers, dropout)
        else:
            raise ValueError('Unknown GNN type: {}'.format(gnn_type))
    

    def forward(self, API_graph, skill_graph):
        API_X = API_graph.ndata["feat"]
        skill_X = skill_graph.ndata["feat"]


        if self.gnn_type == 'Graphormer':
            API_bias = API_graph.ndata["bias"]
            skill_bias = skill_graph.ndata["bias"]
            API_Emb = self.encoder(API_X, API_bias)
            skill_Emb = self.encoder(skill_X, skill_bias)
            return API_Emb, skill_Emb
        else:
            API_Emb = self.encoder(API_graph, API_X)


            skill_Emb = self.encoder(skill_graph, skill_X)

        
        return API_Emb, skill_Emb
        