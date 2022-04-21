import sys
from supervised_model.sup_model import UnsupEmbedding
sys.path.append('..')
from utils import config as cfg
from utils.modules import MyDense
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
from copy import deepcopy
import math


class PointerAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(PointerAttention, self).__init__()
        self._Wq = nn.Linear(hidden_dim, hidden_dim)
        self._Wk = nn.Linear(hidden_dim, hidden_dim)
        self._v = nn.Linear(hidden_dim, 1, bias=False)
        # self._v = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        # nn.init.uniform_(self._v, a=-0.5, b=0.5)
        self._tanh = nn.Tanh()
        self._softmax = nn.Softmax(dim=1)
    
    def forward(self, cur_embedd, centroids):
        '''
        `cur_embedd`: (B, 1, hidden_dim)
        `centroids`: (B, num_clusters, hidden_dim)
        '''
        # print(centroids)
        #print('cluster_embeddings', cluster_embeddings)
        e = self._tanh(self._Wq(cur_embedd) + self._Wk(centroids))  # B * num_clusters * hidden_dim
        #print(e)
        # V = self._v.unsqueeze(0).expand(cur_embedd.shape[0], -1).unsqueeze(1)  # B * 1 * hidden_dim
        # scores = torch.bmm(V, e.transpose(1, 2))  # out: B * 1 * num_clusters
        # out = scores.squeeze(1)  # out: (B, num_clusters)
        #print(self._v)
        
        #print(scores)
        scores = self._v(e)  # out: B * num_clusters * 1
        out = scores.squeeze(-1)
        
        # out = self._softmax(out)
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
        if self._use_rnn:
            # lstm encode the past embeddings
            if self._mode == 'train':
                # if lens[0, 0, 0] < 128:
                #     # print(lens)
                embedds = pack_padded_sequence(embedds, 
                                            #lengths=lens, 
                                            lengths=lens.reshape(lens.shape[0]),
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
        return q


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
                device,
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
        
        self._device = device
    
    def load_frontend(self, pretrained):
        checkpoint = torch.load(pretrained)
        self._frontend.load_state_dict(checkpoint['state_dict'])
    
    def set_mode(self, mode):
        self._backend._mode = mode

    def get_frontend(self):
        return deepcopy(self._frontend).to(torch.device('cpu'))

    def forward(self, obs, state=None, info={}):
        # print(obs)
        embedds = torch.as_tensor(obs['embedding_space'], device=self._device, dtype=torch.float32)   # (B, max_seq_len, feature)
        cur_chunk = torch.as_tensor(obs['cur_chunk'], device=self._device, dtype=torch.float32)   # (B, chunk_len, bin)
        centroids = torch.as_tensor(obs['centroids'], device=self._device, dtype=torch.float32)   # (B, num_clusters, hidden_dim)
        lens = obs['lens']
        # print(embedds.shape)
        # print(cur_chunk.shape)
        # print(onehot_mask.shape)
        # print('forward on device {}'.format(embedds.device))
        feature = self._frontend(cur_chunk.transpose(-1, -2))
        #print(feature)
        logits = self._backend(feature, embedds, centroids, lens)

        # print(logits)
        # print(torch.argmax(logits, dim=-1))
        # return out
        return logits, state
