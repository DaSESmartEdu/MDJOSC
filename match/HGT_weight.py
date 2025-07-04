import math

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax


class HGTLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        heads,
        dropout=0.2,
        use_norm=False,
    ):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.heads = heads
        self.d_k = out_dim // heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(
            torch.ones(self.num_relations, self.heads)
        )
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                sub_graph = G[srctype, etype, dsttype]

                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                k = k_linear(h[srctype]).view(-1, self.heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.heads, self.d_k)

                e_id = self.edge_dict[etype]

                relation_att = self.relation_att[e_id]
                relation_pri = self.relation_pri[e_id]
                relation_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, relation_att)
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                sub_graph.srcdata["k"] = k
                sub_graph.dstdata["q"] = q
                sub_graph.srcdata["v_%d" % e_id] = v

                sub_graph.apply_edges(fn.v_dot_u("q", "k", "t"))
                attn_score = (
                    sub_graph.edata.pop("t").sum(-1)
                    * relation_pri
                    / self.sqrt_dk
                )
                
                if 'weight' in sub_graph.edata:
                    attn_score = attn_score * sub_graph.edata['weight'].unsqueeze(1)
               
                attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

                sub_graph.edata["t"] = attn_score.unsqueeze(-1)

            G.multi_update_all(
                {
                    etype: (
                        fn.u_mul_e("v_%d" % e_id, "t", "m"),
                        fn.sum("m", "t"),
                    )
                    for etype, e_id in edge_dict.items()
                },
                cross_reducer="mean",
            )

            new_h = {}
            for ntype in G.ntypes:
      
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
                trans_out = self.drop(self.a_linears[n_id](t))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h

class HGT(nn.Module):
    def __init__(
        self,
        G,
        node_dict,
        edge_dict,
        in_size_dict,
        hid_size,
        out_size,
        num_layers,
        heads,
        dropout,
        use_norm=True,
    ):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.in_size_dict = in_size_dict
        self.hid_size = hid_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.adapt_ws = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for ntype, ntype_id in node_dict.items():
            self.adapt_ws.append(nn.Linear(self.in_size_dict[ntype], hid_size))
        for _ in range(num_layers):
            self.gcs.append(
                HGTLayer(
                    hid_size,
                    hid_size,
                    node_dict,
                    edge_dict,
                    heads,
                    dropout,
                    use_norm=use_norm,
                )
            )
        self.out = nn.Linear(hid_size, out_size)

    def forward(self, G):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data["feat"]))
        for i in range(self.num_layers):
            h = self.gcs[i](G, h)
        return h