import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

class GAT_encoder(nn.Module):
    def __init__(self, hid_size, out_size, heads, num_layers, dropout):
        super(GAT_encoder, self).__init__()
        self.gat_layers = nn.ModuleList()
        
        
        for i in range(num_layers):
            if i == 0:
              
                self.gat_layers.append(
                    dglnn.GATConv(hid_size, hid_size, heads[i], activation=F.relu)
                )
            elif i == num_layers - 1:
             
                self.gat_layers.append(
                    dglnn.GATConv(
                        hid_size * heads[i-1],
                        out_size,
                        heads[i],
                        residual=True,
                        activation=None,
                    )
                )
            else:
                
                self.gat_layers.append(
                    dglnn.GATConv(
                        hid_size * heads[i-1],
                        hid_size,
                        heads[i],
                        residual=True,
                        activation=F.relu,
                    )
                )
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs): 
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == len(self.gat_layers) - 1:  
                h = h.mean(1)
            else:  
                h = h.flatten(1)
                h = self.dropout(h)
        return h