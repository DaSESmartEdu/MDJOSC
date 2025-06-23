import argparse



def parse_tuple(string):
    return tuple(item.strip() for item in string.strip("()").split(","))


def parse_args():
    parser = argparse.ArgumentParser(description="Joint Training")
   
    parser.add_argument('--API_dict', type = str, default="../data/dict/API_dict.json")
    parser.add_argument('--skill_dict', type = str, default="../data/dict/skill_dict.json")
    parser.add_argument('--job_dict', type = str, default="../data/dict/job_dict.json")
    parser.add_argument('--user_dict', type = str, default="../data/dict/user_dict.json")
    parser.add_argument('--repo_dict', type = str, default="../data/dict/repo_dict.json")

    
    parser.add_argument('--Align_API_feature', type = str, default="../data/alignment_feature/API_feat.npy")
    parser.add_argument('--Align_skill_feature', type = str, default="../data/alignment_feature/skill_feat.npy")
    parser.add_argument('--observed_link', type = str, default="../data/graph/anchor_link/observed_link/observed_link.csv")
    parser.add_argument('--gnn_type', type=str, default='GCN')
    parser.add_argument('--use_norm', type=bool, default=True)
    parser.add_argument('--k', default=5, type=int, help='numbers of postive and negative samples')
    parser.add_argument('--temperature', default=0.1, type=float)
    parser.add_argument('--Align_in_size', type=int, default=768)

    
    parser.add_argument('--API_feature', type = str, default="../data/feature/API_features.npy")
    parser.add_argument('--skill_feature', type = str, default="../data/feature/skill_features.npy")
    parser.add_argument('--job_feature', type = str, default="../data/feature/job_features.npy")
    parser.add_argument('--user_feature', type = str, default="../data/feature/user_features.npy")
    parser.add_argument('--repo_feature', type = str, default="../data/feature/repo_features.npy")

   
    parser.add_argument('--user_repo_own', type = str, default="../data/graph/github/user_repo_own.csv")
    parser.add_argument('--user_repo_fork', type = str, default="../data/graph/github/user_repo_fork.csv")
    parser.add_argument('--user_repo_star', type = str, default="../data/graph/github/user_repo_star.csv")
    parser.add_argument('--repo_API', type = str, default="../data/graph/github/repo_API.csv")
    parser.add_argument('--API_API', type = str, default="../data/graph/github/API_graph/k-nn_sim/API_graph_kNN_15.csv")

     
    parser.add_argument('--user_job', type = str, default="../data/graph/linkedin/user_job.csv")
    parser.add_argument('--job_skill', type = str, default="../data/graph/linkedin/job_skill.csv")
    parser.add_argument('--job_job', type = str, default="../data/graph/linkedin/job_job.csv")
    parser.add_argument('--skill_skill', type = str, default="../data/graph/linkedin/skill_graph/k-nn_sim/skill_graph_kNN_15.csv")

    
    parser.add_argument('--train_pos', type = str, default="../data/dataset/train_pos.csv")
    parser.add_argument('--val_pos', type = str, default="../data/dataset/val_pos.csv")
    parser.add_argument('--test_pos', type = str, default="../data/dataset/test_pos.csv")
    parser.add_argument('--neg_k', type = int, default=3)
    
   
    parser.add_argument('--API_match_size', type=int, default=200)
    parser.add_argument('--skill_match_size', type=int, default=768)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--out_size', type=int, default=64)

    
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--seed', default=2024, type=int)
    
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=0.5)

    parser.add_argument('--edge_type_dgl', type=parse_tuple, default=('API', 'relation_As', 'skill'), help='Supplementary edge type')
    parser.add_argument('--rev_edge_type_dgl', type=parse_tuple, default=('skill', 'rev_relation_As', 'API'), help='Supplementary edge type')
    
    parser.add_argument('--link_pred_op', type=str, default='cos', choices=['dot', 'cos', 'ele', 'cat'])
    parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('--epochs', type=int, default=200, help='How many epochs to train')
    parser.add_argument('--eval_interval', type=int, default=5, help="Evaluate once per how many epochs")
    parser.add_argument('--patience', type=int, default=5, help="patience")
    parser.add_argument("--clip", type=int, default=1.0)
    parser.add_argument("--dropout", type=int, default=0.2)

    
    parser.add_argument('--save_path', type = str, default="./result/model/Joint_Graphormer_HGT.pth")
    parser.add_argument('--log_path', type = str, default="./result/log/Joint_Graphormer_HGT.log")
    return parser.parse_args()