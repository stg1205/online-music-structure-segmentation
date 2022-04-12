import sys
from supervised_model.sup_model import UnsupEmbedding
sys.path.append('..')
from utils import config as cfg
from utils.modules import MyDense
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch


# class PointerAttention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(PointerAttention, self).__init__()
#         self._Wq = nn.Linear(hidden_dim, hidden_dim)
#         self._Wk = nn.Linear(hidden_dim, hidden_dim)
#         self._v = nn.Linear(hidden_dim, 1, bias=False)
    
#     def forward(self, cluster_embeddings, onehot_mask):
#         '''
#         `cluster_embeddings`: (B, k+1, hidden_dim)
#         `onehot_mask`: (B, k+1, num_clusters)
#         '''
#         #print('cluster_embeddings', cluster_embeddings)
#         e = torch.tanh(self._Wq(cluster_embeddings[:, -1, :]).unsqueeze(dim=1) + self._Wk(cluster_embeddings))  # B * k+1 * hidden_dim
#         #print('e', e)
#         scores = self._v(e).transpose(-1, -2)  # out: B * 1 * k+1
#         #print('scores', scores)
#         cluster_sum_scores = torch.matmul(scores, onehot_mask.float())  # out: (B, 1, num_clusters)
#         #print('cluster_num_scores', cluster_sum_scores)
#         cluster_num = torch.sum(onehot_mask, dim=1, keepdim=True)  # out: (B, 1, num_clusters)
#         #print('cluster_num', cluster_num)
#         out = cluster_sum_scores / cluster_num
#         #print('out', out)
#         out = torch.where(out.isnan(), torch.tensor(0, dtype=torch.float).to(out.device), out)
#         #out = cluster_sum_scores
#         out = out.squeeze(1)  # out: (B, num_clusters)
#         #out = F.softmax(out, dim=1)
#         return out


# class Backend(nn.Module):
#     def __init__(self, input_size, 
#                         hidden_size, 
#                         num_layers, 
#                         num_clusters, 
#                         num_heads,
#                         cluster_encode=True, 
#                         mode='train'):
#         super(Backend, self).__init__()
#         self._lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
#         self._attention = PointerAttention(hidden_dim=hidden_size)
#         self._cluster_encode = cluster_encode
#         self._num_clusters = num_clusters

#     def _encode_new_cluster(self, feature, onehot_mask):
#         # encode current embedding
#         max_cluster_nums = torch.max(torch.argmax(onehot_mask, dim=-1), dim=1)[0]  # (1, B)
#         #print(max_cluster_nums)
#         cluster_encodes = torch.zeros((feature.size(0), self._num_clusters), dtype=torch.long)  # (B, num_clusters)
#         #cluster_encodes = F.one_hot(max_cluster_nums+1, num_classes=self._num_clusters)  # out: (B, num_clusters)
        
#         #print((max_cluster_nums+1 < self._num_clusters).nonzero(as_tuple=True))
#         not_full_idxs = (max_cluster_nums+1 < self._num_clusters).nonzero(as_tuple=True)[0]
#         cluster_encodes.index_put_(indices=(not_full_idxs, (max_cluster_nums+1)[not_full_idxs]), values=torch.tensor(1))

#         return cluster_encodes

#     def forward(self, feature, embedds, onehot_mask, padded):
#         '''
#         feature: (B, emb_dim)
#         embedds: (B, max_seq_len, emb_dim) padding
#         one-hot mask: (B, max_seq_len, num_clusters) eg. [(0, 0, 1, 0, 0, 0, 1, ...), ...]
#         '''
#         #print(onehot_mask.shape)
#         # lstm encode the past embeddings
#         # if isinstance(embedds, PackedSequence):
#         #     print('embedds', embedds)
#         #     print(pad_packed_sequence(embedds, batch_first=True)[0].isnan().sum())
#         lstm_embedds, (hn, cn) = self._lstm(embedds)  
#         if isinstance(lstm_embedds, PackedSequence):
#             print('lstm_embedds', lstm_embedds)
#         if padded:
#             lstm_embedds, out_len = pad_packed_sequence(lstm_embedds, batch_first=True)  # out: (batch, max_seq_len, hidden_size), hn: (batch, num_layers, hidden_size)

#         # encode the cluster of current embedding
#         cluster_encodes = self._encode_new_cluster(feature, onehot_mask).to(feature.device)
#         #print(cluster_encodes)
#         if self._cluster_encode:
#             feature = torch.cat([feature, cluster_encodes], dim=1)
#         onehot_mask = torch.cat([onehot_mask, cluster_encodes.unsqueeze(1)], dim=1)

#         # lstm encode current feature using the previous output
#         lstm_cur_embedd, (h_cur, c_cur) = self._lstm(feature.unsqueeze(1), (hn, cn))
#         lstm_embedds = torch.cat([lstm_embedds, lstm_cur_embedd], dim=1)
#         #print(lstm_embedds.shape)
#         #print(onehot_mask.shape)
#         #print()
#         attention_out = self._attention(lstm_embedds, onehot_mask) 
#         #q = torch.sum(attention_out, dim=2)
#         return attention_out

class PointerAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(PointerAttention, self).__init__()
        self._Wq = nn.Linear(hidden_dim, hidden_dim)
        self._Wk = nn.Linear(hidden_dim, hidden_dim)
        self._v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, cur_embedd, centroids):
        '''
        `cur_embedd`: (B, 1, hidden_dim)
        `centroids`: (B, num_clusters, hidden_dim)
        '''
        #print('cluster_embeddings', cluster_embeddings)
        e = torch.tanh(self._Wq(cur_embedd) + self._Wk(centroids))  # B * num_clusters * hidden_dim
        scores = self._v(e)  # out: B * num_clusters * 1
        out = scores.squeeze(-1)  # out: (B, num_clusters)
        #out = F.softmax(out, dim=1)
        # print(out)
        return out
    
class Backend(nn.Module):
    def __init__(self, input_size, 
                        hidden_size, 
                        num_layers, 
                        num_clusters, 
                        num_heads,
                        mode,
                        use_rnn,
                        cluster_encode=True):
        super(Backend, self).__init__()
        self._lstm = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=num_layers, batch_first=True)
        self._attention = PointerAttention(hidden_dim=input_size)
        self._cluster_encode = cluster_encode
        self._num_clusters = num_clusters
        self._use_rnn = use_rnn
        self._mode = mode

    def forward(self, feature, embedds, centroids, lens):
        '''
        feature: (B, emb_dim)
        embedds: (B, max_seq_len, emb_dim) padding
        `centroids`: (B, num_clusters, hidden_dim)
        '''
        #print(onehot_mask.shape)
        if self._use_rnn:
            # lstm encode the past embeddings
            if self._mode == 'train':
                embedds = pack_padded_sequence(embedds, 
                                            lengths=lens, 
                                            batch_first=True,
                                            enforce_sorted=False)
            lstm_embedds, (hn, cn) = self._lstm(embedds) 

            if self._mode == 'train':
                ori_idxs = embedds.unsorted_indices
                # back to original batch index
                hn, cn = hn[:, ori_idxs, :], cn[:, ori_idxs, :]

        # # if self._mode == 'train':
        # #     lstm_embedds, out_len = pad_packed_sequence(lstm_embedds, batch_first=True)  # out: (batch, max_seq_len, hidden_size), hn: (batch, num_layers, hidden_size)

        if self._cluster_encode:
            # encode the cluster of current embedding
            zero_encodes = torch.zeros([feature.size(0), self._num_clusters]).to(feature.device)
            feature = torch.cat([feature, zero_encodes], dim=1)

        # #print(feature)
        # # lstm encode current feature using the previous output
        cur_cluster_embedd = feature.unsqueeze(1)
        if self._use_rnn:
            cur_cluster_embedd, (h_cur, c_cur) = self._lstm(cur_cluster_embedd, (hn, cn))
        
        q = self._attention(cur_cluster_embedd, centroids) 
        return q, cur_cluster_embedd


class QNet(nn.Module):
    def __init__(self, 
                input_shape, 
                embedding_size, 
                hidden_size,
                num_layers, 
                num_heads,
                num_clusters,
                cluster_encode,
                freeze_frontend,
                use_rnn,
                mode='train'):
        super(QNet, self).__init__()
        self._frontend = UnsupEmbedding(input_shape)
        if freeze_frontend:
            self._frontend.requires_grad_(False)
        self._backend = Backend(input_size=embedding_size, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers, 
                                num_clusters=num_clusters, 
                                cluster_encode=cluster_encode,
                                num_heads=num_heads,
                                use_rnn=use_rnn,
                                mode=mode)
    
    def load_frontend(self, pretrained):
        checkpoint = torch.load(pretrained)
        self._frontend.load_state_dict(checkpoint['state_dict'])
    
    def set_mode(self, mode):
        self._backend._mode = mode

    def get_frontend(self):
        return self._frontend

    def forward(self, embedds, cur_chunk, centroids, lens=None):
        
        # embedds = state['embedding_space']  # (B, max_seq_len, feature)
        # cur_chunk = state['cur_chunk']  # (B, chunk_len, bin)
        # centroids = state['centroids']  # (B, num_clusters, hidden_dim)
        # print(embedds.shape)
        # print(cur_chunk.shape)
        # print(onehot_mask.shape)
        # print('forward on device {}'.format(embedds.device))
        feature = self._frontend(cur_chunk)
        q, cur_cluster_embedd = self._backend(feature, embedds, centroids, lens)

        out = {
            'q': q,
            'cur_cluster_embedd': cur_cluster_embedd,
            'new_feature': feature
        }

        return out
