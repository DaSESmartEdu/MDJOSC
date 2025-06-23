import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import codecschmod a+x hfd.sh
import json
from tqdm import tqdm


class jobTitleNormalizer:
    def __init__(self, MODEL_PATH, CLS_flag):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.CLS_flag = CLS_flag
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModel.from_pretrained(MODEL_PATH).to(self.device)
    
    def get_embedding(self, title):
        indexed_title = self.tokenizer.encode(title, max_length=20, truncation=True) 
        tokens_tensor = torch.tensor([indexed_title]).to(self.device)  
        with torch.no_grad():
            outputs = self.model(tokens_tensor)

        if self.CLS_flag:
           title_embeddings = outputs.pooler_output  
        else:
           title_embeddings = torch.mean(outputs.last_hidden_state, dim=1)
        title_embeddings = title_embeddings.detach().cpu().numpy()  
        return title_embeddings
    
    def get_standard_embedding(self, standard_titles):
        standard_embeddings = np.concatenate([self.get_embedding(title) for title in standard_titles], axis=0)
        return standard_embeddings

    def match(self, job_titles, standard_embeddings, standard_titles):
        best_matches = []
        for title in tqdm(job_titles):
            title_embedding = self.get_embedding(title)
            similities = cosine_similarity(title_embedding, standard_embeddings)
            best_match_index = np.argmax(similities)
            best_matches.append(standard_titles[best_match_index])
        return best_matches

MODEL_PATH = './model_1'
CLS_flag = False
job_title_normalizer = jobTitleNormalizer(MODEL_PATH, CLS_flag)

job_data = pd.read_excel('./s_title.xlsx')
standard_title = job_data['s_title']
s_title = standard_title.to_list()
s_title = [title.strip() for title in s_title]

standard_embeddings = job_title_normalizer.get_standard_embedding(s_title)

def match_high_title(s_titles, job_data):

    title_mapping = dict(zip(job_data['s_title'], job_data['cs_title']))
    h_s_titles = [title_mapping.get(title) for title in s_titles]
    return h_s_titles 