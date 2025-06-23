import torch
import torch.nn as nn
import torch.nn.functional as F
from alignment.model import alignment_model
from alignment.loss import unsupervised_loss, supervised_loss
from match.HGT_weight import HGT
from dgl.nn import  EdgePredictor

class node_match(nn.Module):
    def __init__(self, hid_size, out_size, src_ntype, tgt_ntype, link_pred_op):
        super(node_match, self).__init__()
        self.predictor = EdgePredictor(op=link_pred_op)
        self.src_ntype = src_ntype
        self.tgt_ntype = tgt_ntype
        
    def forward(self, node_embeddings, node_nids):

        src_h = node_embeddings[self.src_ntype][node_nids[0]]
       
        tgt_h = node_embeddings[self.tgt_ntype][node_nids[1]]
    
        score = self.predictor(src_h, tgt_h).view(-1)
        
        return score, src_h, tgt_h

class JointModel(nn.Module):
    def __init__(self, My_graph, node_dict, edge_dict, in_size_dict, device, args):
        super(JointModel, self).__init__()
        self.alignment_model = alignment_model(args.gnn_type, 
                                               args.Align_in_size,
                                               args.hid_size,
                                               args.out_size,
                                               args.heads,
                                               args.num_layers,
                                               args.dropout
                                               ).to(device)
        self.hg_model = HGT(My_graph, node_dict, edge_dict,
                            in_size_dict, args.hid_size, args.out_size,
                            args.num_layers, args.heads, args.dropout, use_norm=args.use_norm).to(device)
        
        self.node_match = node_match(args.hid_size, args.out_size, 'user', 'job', 'cos')
        self.to(device)
        
    def similarity_matrix(self, api_emb, skill_emb):
        matrix = F.cosine_similarity(api_emb.unsqueeze(1), skill_emb.unsqueeze(0), dim=-1)

        return matrix

    def supp_my_graph(self, alignment_matrix, My_graph, etype):
        src, tgt = My_graph.edges(etype=etype)
        if etype == ('API', 'relation_As', 'skill'):
            edge_weights = alignment_matrix[src, tgt]
        elif etype == ('skill', 'rev_relation_As', 'API'):
            edge_weights = alignment_matrix.T[src, tgt] 
        My_graph.edges[etype].data['weight'] = edge_weights
       
        return My_graph
    def forward(self, API_graph, skill_graph, My_graph, data_pos, data_neg, args):
        
       
        API_Emb, skill_Emb = self.alignment_model(API_graph, skill_graph)
        align_matrix = self.similarity_matrix(API_Emb, skill_Emb)
        
        My_graph = self.supp_my_graph(align_matrix, My_graph, args.edge_type_dgl)
        My_graph = self.supp_my_graph(align_matrix, My_graph, args.rev_edge_type_dgl)
        
        
        node_embeddings = self.hg_model(My_graph)
        
        pos_score, src_pos_h, tgt_pos_h = self.node_match(node_embeddings, data_pos)
        neg_score, src_neg_h, tgt_neg_h = self.node_match(node_embeddings, data_neg)
        return API_Emb, skill_Emb, pos_score, neg_score, align_matrix, node_embeddings


def joint_loss(API_Emb, skill_Emb, observed_links, align_matrix, pos_score, neg_score, args):

   
    s_loss = supervised_loss(API_Emb, skill_Emb, observed_links)
    u_loss = unsupervised_loss(API_Emb, skill_Emb, align_matrix, args.k, args.temperature)
    align_loss = args.lambda1 * s_loss + args.lambda2 * u_loss
    
    
    scores = torch.cat([pos_score, neg_score])
    device = scores.device
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)

    pos_weight = neg_score.shape[0] / pos_score.shape[0]
    match_loss = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=torch.tensor(pos_weight).to(device))
    
    loss = align_loss + match_loss
    return loss