import pandas as pd
import numpy as np
import random

import json
from tqdm import tqdm
from args import parse_args

import torch
import dgl
from dgl import AddSelfLoop
from dgl import AddReverse

import sys
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def Load_Dict(dict_path):
    with open(dict_path, 'r') as f:
        data_dict = json.load(f)
    data_dict = {key: int(value) for key, value in data_dict.items()}
    return data_dict

def Get_Dicts(dict_paths):
    node_dicts = {node_type: Load_Dict(dict_path) 
                  for node_type, dict_path in dict_paths.items()}
    return node_dicts


def Load_Feature(feature_path):
    
    feature = np.load(feature_path)
    feature = torch.from_numpy(feature).to(torch.float32)

    return feature

def Get_Features(feature_paths):
    node_features = {node_type: Load_Feature(feature_path) 
                  for node_type, feature_path in feature_paths.items()}
    return node_features

def str2id(data, dicts):
    user = torch.tensor(data['user'].map(dicts['user']).values, dtype=torch.long)
    job = torch.tensor(data['job'].map(dicts['job']).values, dtype=torch.long)
    new_data = (user, job)
    return new_data

def Get_Dataset(dataset_paths, dicts, k):
    datasets = {}

    for key, path in dataset_paths.items():
        data_pos = pd.read_csv(path, dtype=str)
        data_pos = data_pos.drop_duplicates()
        datasets[key] = data_pos

        neg_key = key.replace('pos', 'neg')
        data_neg = negative_sample(data_pos, dicts['job'], k)
        datasets[neg_key] = data_neg

    datasets_id = {dataset_type: str2id(dataset, dicts)
                 for dataset_type, dataset in datasets.items()}

    return datasets_id

def negative_sample(dataset_pos, job_dict, k):

    negative_samples = []
    for _, row in dataset_pos.iterrows():
        user = row['user']
        pos_job = row['job']
        neg_jobs = [job for job in job_dict if job != pos_job]
        neg_samples = random.sample(neg_jobs, k)
        for neg_job in neg_samples:
            negative_samples.append((user, neg_job))
    negative_df = pd.DataFrame(negative_samples, columns=['user', 'job'])
    return negative_df


def get_bias(graph, args):
    num_nodes = graph.num_nodes()
    attn_bias = torch.zeros((1, num_nodes, num_nodes, args.heads))
    adj_matrix = graph.adj().to_dense()
    attn_bias[0, :, :, :] = float('-inf')

    nonzero_indices = adj_matrix.nonzero(as_tuple=True)
    for i, j in zip(nonzero_indices[0], nonzero_indices[1]):
        attn_bias[0, i, j, :] = 0.0
    return attn_bias
    
def get_observed_links(observed_path, dicts):

    groundtruth = pd.read_csv(observed_path)
    groundtruth = groundtruth.dropna()

    API_list = groundtruth['API'].tolist()
    skill_list = groundtruth['skill'].tolist()

    API_id = [dicts['API'][item] for item in API_list]
    skill_id = [dicts['skill'][item] for item in skill_list]

    observed_links = []
    for API, skill in zip(API_id, skill_id):
        observed_links.append((API, skill))
    observed_links = torch.tensor(observed_links)
    observed_links = observed_links
    return observed_links

def Get_Graphdata(graph_path):
    graph_data = pd.read_csv(graph_path, dtype=str)
    graph_data = graph_data.dropna()
    graph_data = graph_data.drop_duplicates()
    return graph_data




def add_edge_weights_dgl(G, edge_weights):
    for etype in G.canonical_etypes:
        if etype in edge_weights:
            G.edges[etype].data['weight'] = edge_weights[etype]
        else:
            num_edges = G.number_of_edges(etype)
            G.edges[etype].data['weight'] = torch.ones(num_edges, dtype=torch.float32)
    
    return G

def hetero_dgl_graph(graph_paths, node_dicts):
    
    node_types = ['user', 'repo', 'job', 'API', 'skill']
    edge_types = [('user', 'own', 'repo'), 
                  ('user', 'fork', 'repo'),
                  ('user', 'star', 'repo'),  
                  ('repo', 'contain', 'API'), 
                  ('user', 'work', 'job'),
                  ('API', 'relation_A', 'API'), 
                  ('job', 'require', 'skill'), 
                  ('job', 'shift', 'job'),
                  ('skill', 'relation_s', 'skill'),
                  ('API', 'relation_As', 'skill')
                ]
    
    hetero_g = dgl.heterograph({
        edge_type: ([], []) for edge_type in edge_types
    })
    
    user_repo_own = Get_Graphdata(graph_paths['user_repo_own'])
    user_repo_fork = Get_Graphdata(graph_paths['user_repo_fork'])
    user_repo_star = Get_Graphdata(graph_paths['user_repo_star'])
    repo_API = Get_Graphdata(graph_paths['repo_API'])
    API_API = Get_Graphdata(graph_paths['API_API'])
    user_job = Get_Graphdata(graph_paths['user_job'])
    job_skill = Get_Graphdata(graph_paths['job_skill'])
    skill_skill = Get_Graphdata(graph_paths['skill_skill'])
    job_job = Get_Graphdata(graph_paths['job_job'])

    user_repo_own['user'] = user_repo_own['user'].map(node_dicts['user'])
    user_repo_own['repo'] = user_repo_own['repo'].map(node_dicts['repo'])
    user_repo_fork['user'] = user_repo_fork['user'].map(node_dicts['user'])
    user_repo_fork['repo'] = user_repo_fork['repo'].map(node_dicts['repo'])
    user_repo_star['user'] = user_repo_star['user'].map(node_dicts['user'])
    user_repo_star['repo'] = user_repo_star['repo'].map(node_dicts['repo'])

    repo_API['repo'] = repo_API['repo'].map(node_dicts['repo'])
    repo_API['API'] = repo_API['API'].map(node_dicts['API'])

    API_API['API_1'] = API_API['API_1'].map(node_dicts['API'])
    API_API['API_2'] = API_API['API_2'].map(node_dicts['API'])

    skill_skill['skill_1'] = skill_skill['skill_1'].map(node_dicts['skill'])
    skill_skill['skill_2'] = skill_skill['skill_2'].map(node_dicts['skill'])

    job_skill['job'] = job_skill['job'].map(node_dicts['job'])
    job_skill['skill'] = job_skill['skill'].map(node_dicts['skill'])

    job_job['job_1'] = job_job['job_1'].map(node_dicts['job'])
    job_job['job_2'] = job_job['job_2'].map(node_dicts['job'])

    user_job['user'] = user_job['user'].map(node_dicts['user'])
    user_job['job'] = user_job['job'].map(node_dicts['job'])

  
    user_repo_own_index = torch.tensor(user_repo_own[['user', 'repo']].values).t().contiguous()

    user_repo_fork_index = torch.tensor(user_repo_fork[['user', 'repo']].values).t().contiguous()

    user_repo_star_index = torch.tensor(user_repo_star[['user', 'repo']].values).t().contiguous()

    repo_API_index = torch.tensor(repo_API[['repo', 'API']].values).t().contiguous()

    API_API_index = torch.tensor(API_API[['API_1', 'API_2']].values).t().contiguous()


    skill_skill_index = torch.tensor(skill_skill[['skill_1', 'skill_2']].values).t().contiguous()



    user_job_index = torch.tensor(user_job[['user', 'job']].values).t().contiguous()


    job_skill_index = torch.tensor(job_skill[['job', 'skill']].values).t().contiguous()


    job_job_index = torch.tensor(job_job[['job_1', 'job_2']].values).t().contiguous()

    
    edge_data = {
        ('user', 'own', 'repo'): (user_repo_own_index[0], user_repo_own_index[1]),
        ('user', 'fork', 'repo'): (user_repo_fork_index[0], user_repo_fork_index[1]),
        ('user', 'star', 'repo'): (user_repo_star_index[0], user_repo_star_index[1]),
        ('repo', 'contain', 'API'): (repo_API_index[0], repo_API_index[1]),
        ('user', 'work', 'job'): (user_job_index[0], user_job_index[1]),
        ('API', 'relation_A', 'API'): (API_API_index[0], API_API_index[1]),
        ('skill', 'relation_s', 'skill'): (skill_skill_index[0], skill_skill_index[1]),
        ('job', 'require', 'skill'): (job_skill_index[0], job_skill_index[1]),
        ('job', 'shift', 'job'): (job_job_index[0], job_job_index[1]),
    }
    
    for edge_type, (src, dst) in edge_data.items():
        hetero_g.add_edges(src, dst, etype=edge_type)
    
    API_nodes = torch.arange(hetero_g.num_nodes('API'))
    skill_nodes = torch.arange(hetero_g.num_nodes('skill'))
    API_skill_index = torch.cartesian_prod(API_nodes, skill_nodes).t().contiguous()
    hetero_g.add_edges(API_skill_index[0], API_skill_index[1], etype=('API', 'relation_As', 'skill'))
    
    return hetero_g

def mask_edges_dgl(hetero_graph, src_ids, tgt_ids, edge_type):
    hg_filtered = hetero_graph.clone()


    src, tgt = hetero_graph.edges(etype=edge_type)
    mask = torch.zeros(len(src), dtype=torch.bool)
    for i in range(len(src_ids)):

        mask = mask | ((src == src_ids[i]) & (tgt == tgt_ids[i]))
    

    hg_filtered.remove_edges(torch.nonzero(~mask).squeeze(), etype=edge_type)
        
    return hg_filtered

   

if __name__ == '__main__':

    args = parse_args()

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

    print('*'*100)
    dataset_paths = {
        'train_pos': args.train_pos,
        'val_pos': args.val_pos,
        'test_pos': args.test_pos
    }
    datasets = Get_Dataset(dataset_paths, dicts, args.neg_k, device)
    user_pos, job_pos = datasets['train_pos']
    
    edge_type = ('user', 'work', 'job')
    My_Graph_dgl = mask_edges_dgl(My_Graph_dgl, user_pos, job_pos, edge_type, device)

   

    transform_dgl = AddReverse()
    My_Graph_dgl = transform_dgl(My_Graph_dgl)

    
    weights = {}
    My_Graph_dgl = add_edge_weights_dgl(My_Graph_dgl, weights, device)
    print(My_Graph_dgl)
    print('*'*100)
    
    
    
    
    
    


