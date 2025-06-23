import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphormerLayer
import torch.nn.functional as F


class Graphormer_encoder(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads, num_layers, dropout):
        super().__init__()
        self.gt_layers = nn.ModuleList()
        
        
        self.gt_layers.append(
            GraphormerLayer(
                in_size,
                hid_size,
                heads,
            )
        )

        for l in range(1, num_layers - 1):
            self.gt_layers.append(
                GraphormerLayer(
                    in_size, 
                    hid_size,
                    heads,  
                )
            )
        
       
        self.fc = nn.Linear(in_size, out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, bias=None):
        
        x, bias = features.unsqueeze(0), bias.unsqueeze(0)
        h = self.dropout(x)
       
        for i, layer in enumerate(self.gt_layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, bias)
            h = F.relu(h)
        h = self.fc(h)
        h = h.squeeze(0)
        return self.dropout(h)