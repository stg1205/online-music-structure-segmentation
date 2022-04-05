import sys
sys.path.append('..')
from utils import config as cfg
from utils.modules import MyDense, PointerAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
import torch


class Backend(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_clusters, cluster_encode=True, mode='train'):
        super(Backend, self).__init__()
        self._lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self._attention = PointerAttention(hidden_dim=hidden_size)
        self._cluster_encode = cluster_encode
        self._num_clusters = num_clusters

    def _encode_new_cluster(self, feature, onehot_mask):
        # encode current embedding
        max_cluster_nums = torch.max(torch.argmax(onehot_mask, dim=-1), dim=1)[0]  # (1, B)
        print(max_cluster_nums)
        cluster_encodes = torch.zeros((feature.size(0), self._num_clusters), dtype=torch.long)  # (B, num_clusters)
        #cluster_encodes = F.one_hot(max_cluster_nums+1, num_classes=self._num_clusters)  # out: (B, num_clusters)
        
        print((max_cluster_nums+1 < self._num_clusters).nonzero(as_tuple=True))
        not_full_idxs = (max_cluster_nums+1 < self._num_clusters).nonzero(as_tuple=True)[0]
        cluster_encodes.index_put_(indices=(not_full_idxs, (max_cluster_nums+1)[not_full_idxs]), values=torch.tensor(1))

        return cluster_encodes

    def forward(self, feature, embedds, onehot_mask, padded):
        '''
        feature: (B, emb_dim)
        embedds: (B, max_seq_len, emb_dim) padding
        one-hot mask: (B, max_seq_len, num_clusters) eg. [(0, 0, 1, 0, 0, 0, 1, ...), ...]
        '''
        print(onehot_mask.shape)
        # lstm encode the past embeddings
        lstm_embedds, (hn, cn) = self._lstm(embedds)  
        if padded:
            lstm_embedds, out_len = pad_packed_sequence(lstm_embedds, batch_first=True)  # out: (batch, max_seq_len, hidden_size), hn: (batch, num_layers, hidden_size)

        # encode the cluster of current embedding
        cluster_encodes = self._encode_new_cluster(feature, onehot_mask).to(feature.device)
        print(cluster_encodes)
        if self._cluster_encode:
            feature = torch.cat([feature, cluster_encodes], dim=1)
        onehot_mask = torch.cat([onehot_mask, cluster_encodes.unsqueeze(1)], dim=1)

        # lstm encode current feature using the previous output
        lstm_cur_embedd, (h_cur, c_cur) = self._lstm(feature.unsqueeze(1), (hn, cn))
        lstm_embedds = torch.cat([lstm_embedds, lstm_cur_embedd], dim=1)
        print(lstm_embedds.shape)
        print(onehot_mask.shape)
        print()
        attention_out = self._attention(lstm_embedds, onehot_mask) 
        #q = torch.sum(attention_out, dim=2)
        return attention_out


class QNet(nn.Module):
    def __init__(self, frontend, backend):
        super(QNet, self).__init__()
        self._frontend = frontend
        self._backend = backend
    
    def load_frontend(self, pretrained):
        checkpoint = torch.load(pretrained)
        self._frontend.load_state_dict(checkpoint['state_dict'])
    
    def get_frontend(self):
        return self._frontend

    def forward(self, state, padded=True):
        embedds = state['embedding_space']  # (B, max_seq_len, feature)
        cur_chunk = state['cur_chunk']  # (B, chunk_len, bin)
        onehot_mask = state['onehot_mask']  # (B, max_seq_len+1, num_clusters)
        #centroids = state['centroids']
        # print(embedds.shape)
        # print(cur_chunk.shape)
        # print(onehot_mask.shape)
        feature = self._frontend(cur_chunk)
        q = self._backend(feature, embedds, onehot_mask, padded)

        out = {
            'q': q,
            'new_feature': feature
        }

        return out
