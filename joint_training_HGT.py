import logging
import args
from tqdm import tqdm

from utils import Get_Dicts, Get_Features, hetero_dgl_graph, Get_Dataset, mask_edges_dgl, add_edge_weights_dgl, get_observed_links, set_seed,get_bias

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dgl import AddSelfLoop
from dgl import AddReverse
from torch.nn import DataParallel
import torch
import torch.nn.functional as F
from joint_model_HGT import joint_loss, JointModel

import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
def metric(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu()
    preds = (scores > 0.5).float()
    auc = roc_auc_score(labels, scores)
    acc = accuracy_score(labels, preds)
    prec =  precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return auc, acc, prec, rec, f1 

def train(model,
        API_graph,
        skill_graph, 
        My_graph, 
        datasets,
        observed_links,
        optimizer, 
        scheduler,
        device,
        args,
    ):
    set_seed(args.seed)
    best_val_loss = float('inf')
    patience_counter = 0

    train_pos = datasets['train_pos']
    train_neg = datasets['train_neg']
    val_pos = datasets['val_pos']
    val_neg = datasets['val_neg']
    test_pos = datasets['test_pos']
    test_neg = datasets['test_neg']

    for epoch in tqdm(range(1, args.epochs+1)):
        model.train()
        API_Emb, skill_Emb, train_pos_score, train_neg_score, align_matrix, node_embeddings = model(API_graph, 
                                                                                                    skill_graph, 
                                                                                                    My_graph, 
                                                                                                    train_pos, 
                                                                                                    train_neg,
                                                                                                    args
                                                                                                    ) 
    
        train_loss = joint_loss(API_Emb, skill_Emb, observed_links, align_matrix, train_pos_score, train_neg_score, args)
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                model.eval()
                train_auc, train_acc, train_prec, train_rec, train_f1 = metric(train_pos_score, train_neg_score)
                API_Emb, skill_Emb, val_pos_score, val_neg_score, align_matrix, node_embeddings = model(API_graph, 
                                                                                               skill_graph,  
                                                                                               My_graph, 
                                                                                               train_pos, 
                                                                                               train_neg,
                                                                                               args
                                                                                               ) 
                val_loss = joint_loss(API_Emb, skill_Emb, observed_links, align_matrix, val_pos_score, val_neg_score, args)
                val_auc, val_acc, val_prec, val_rec, val_f1 = metric(val_pos_score, val_neg_score)
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, val_auc: {val_auc:.4f}, val_acc:{val_acc:.4f}, val_prec:{val_prec:.4f}, val_rec:{val_rec:.4f},val_f1:{val_f1:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), args.save_path)
                    logger.info(f"Model saved to {args.save_path}")
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
    test_auc, test_acc, test_prec, test_rec, test_f1 = evaluate(model, API_graph, skill_graph, My_graph, test_pos,  test_neg, args)
    print(f"Test auc: {test_auc:.4f}, Test acc:{test_acc:.4f}, Test prec:{test_prec:.4f}, Test rec:{test_rec:.4f}, Test f1:{test_f1:.4f}")
    logger.info(f"Test auc: {test_auc:.4f}, Test acc:{test_acc:.4f}, Test prec:{test_prec:.4f}, Test rec:{test_rec:.4f}, Test f1:{test_f1:.4f}")

def evaluate(model, API_graph, skill_graph, My_graph, test_pos, test_neg, args):
    model.load_state_dict(args.save_path)
    with torch.no_grad():
        model.eval()
        API_Emb, skill_Emb, test_pos_score, test_neg_score, align_matrix, node_embeddings = model(API_graph, 
                                                                                                skill_graph,  
                                                                                                My_graph, 
                                                                                                test_pos, 
                                                                                                test_neg,
                                                                                                args
                                                                                                ) 
        test_metric = metric(test_pos_score, test_neg_score)

    return test_metric


if __name__ == '__main__':

    args = args.parse_args()

    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    logger = logging.getLogger()

    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device('cuda', index=args.device)
    else:
        device = torch.device('cpu')
 
    dict_paths = {
        'user': args.user_dict,
        'job': args.job_dict,
        'repo': args.repo_dict,
        'API': args.API_dict,
        'skill': args.skill_dict
    }
    dicts = Get_Dicts(dict_paths)

    align_feature_paths = {
        'API': args.Align_API_feature,
        'skill': args.Align_skill_feature
    }
    align_features = Get_Features(align_feature_paths)
    
    observed_links = get_observed_links(args.observed_link, dicts)

    feature_paths = {
        'user': args.user_feature,
        'job': args.job_feature,
        'repo': args.repo_feature,
        'API': args.API_feature,
        'skill': args.skill_feature
    }
    features = Get_Features(feature_paths)

    h_graph_paths = {
        'user_repo_own': args.user_repo_own,
        'user_repo_fork': args.user_repo_fork,
        'user_repo_star': args.user_repo_star,
        'repo_API': args.repo_API,
        'API_API': args.API_API,
        'user_job': args.user_job,
        'job_skill': args.job_skill,
        'job_job': args.job_job,
        'skill_skill': args.skill_skill
    }

    My_Graph_dgl = hetero_dgl_graph(h_graph_paths, dicts)
    
    dataset_paths = {
        'train_pos': args.train_pos,
        'val_pos': args.val_pos,
        'test_pos': args.test_pos
    }
    datasets = Get_Dataset(dataset_paths, dicts, args.neg_k)
    user_pos, job_pos = datasets['train_pos']

  
    edge_type = ('user', 'work', 'job')
    My_Graph_dgl = mask_edges_dgl(My_Graph_dgl, user_pos, job_pos, edge_type)
    
 
    API_edge_type = ('API', 'relation_A', 'API')
    skill_edge_type = ('skill', 'relation_s', 'skill')
    API_graph = My_Graph_dgl.edge_type_subgraph([API_edge_type])
    skill_graph = My_Graph_dgl.edge_type_subgraph([skill_edge_type])
    transform = AddSelfLoop()
    API_graph = transform(API_graph)
    skill_graph = transform(skill_graph)

  
    transform_dgl = AddReverse()
    My_Graph_dgl = transform_dgl(My_Graph_dgl)
    API_graph = transform_dgl(API_graph)
    skill_graph = transform_dgl(skill_graph)
    
    API_bias = get_bias(API_graph, args)
    API_graph.ndata['bias'] = API_bias.squeeze(0)
    skill_bias = get_bias(skill_graph, args)
    skill_graph.ndata['bias'] = skill_bias.squeeze(0)

    
    weights = {}
    My_Graph_dgl = add_edge_weights_dgl(My_Graph_dgl, weights)

    node_dict = {} 
    edge_dict = {} 
    for ntype in My_Graph_dgl.ntypes:
        node_dict[ntype] = len(node_dict)

    for etype in My_Graph_dgl.etypes:
        edge_dict[etype] = len(edge_dict)
        My_Graph_dgl.edges[etype].data["id"] = (
            torch.ones(My_Graph_dgl.num_edges(etype), dtype=torch.long) * edge_dict[etype]
        )
    
    in_size_dict = {} 

    for ntype in My_Graph_dgl.ntypes:
        in_size_dict[ntype] = features[ntype].size(-1)
        emb = features[ntype]
        My_Graph_dgl.nodes[ntype].data["feat"] = emb
   
    My_Graph_dgl = My_Graph_dgl.to(device)
    model = JointModel(My_Graph_dgl,
                       node_dict,
                       edge_dict,
                       in_size_dict,
                       device, 
                       args
                    )
 
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, total_steps=args.epochs, max_lr=args.lr * 10
    )
    API_graph = API_graph.to(device)
    skill_graph = skill_graph.to(device)
    API_graph.ndata['feat'] = align_features['API'].to(device)
    skill_graph.ndata['feat'] = align_features['skill'].to(device)
    torch.cuda.empty_cache()
    train(model, API_graph, skill_graph, My_Graph_dgl, datasets, observed_links, 
        optimizer, scheduler, device, args)

    