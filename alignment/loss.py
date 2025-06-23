import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import cosine_similarity




def supervised_loss(api_emb, skill_emb, links):

    api_indices = torch.tensor([link[0] for link in links], dtype=torch.long)
    skill_indices = torch.tensor([link[1] for link in links], dtype=torch.long)

    api_vectors = api_emb[api_indices]
    skill_vectors = skill_emb[skill_indices]

    loss = F.mse_loss(api_vectors, skill_vectors, reduction='mean')
    
    return loss

def unsupervised_loss(emb_API, emb_skill, sim_matric, top_k, temperature):
    
    top_k_positives = torch.topk(sim_matric, top_k, dim=1).indices 
    top_k_negatives = torch.topk(-sim_matric, top_k, dim=1).indices

  
    top_k_positives_skill = torch.topk(sim_matric.t(), top_k, dim=1).indices 
    top_k_negatives_skill = torch.topk(-sim_matric.t(), top_k, dim=1).indices

    def info_nce_loss(query_emb, key_emb, positive_indices, negative_indices):
      
       
        positive_samples = key_emb[positive_indices] 
        negative_samples = key_emb[negative_indices] 

       
        query = query_emb.unsqueeze(1)  
        pos_sim = torch.bmm(query, positive_samples.transpose(1, 2)).squeeze(1)  
        neg_sim = torch.bmm(query, negative_samples.transpose(1, 2)).squeeze(1)  

        
        logits = torch.cat([pos_sim, neg_sim], dim=1)  
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=query_emb.device)  

        logits /= temperature

     
        loss = F.cross_entropy(logits, labels)

        return loss
    
    loss_api_to_skill = info_nce_loss(emb_API, emb_skill, top_k_positives, top_k_negatives)
    loss_skill_to_api = info_nce_loss(emb_skill, emb_API, top_k_positives_skill, top_k_negatives_skill)
    loss = loss_api_to_skill + loss_skill_to_api
    return loss