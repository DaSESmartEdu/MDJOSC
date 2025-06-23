import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

class GCN_encoder(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers, dropout):
        super(GCN_encoder,self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        
      
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.GraphConv(hid_size, hid_size, activation=F.relu))
        
        self.layers.append(dglnn.GraphConv(hid_size, out_size, activation=F.relu))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features): 
        h = self.dropout(features)    
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h