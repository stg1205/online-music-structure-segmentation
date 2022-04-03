import sys
sys.path.append('..')
from utils import config as cfg
from utils.modules import MyDense, PointerAttention
from supervised_model.sup_model import UnsupEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence
import numpy as np
import torch
from collections import deque


class Backend(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, mode='train'):
        super(Backend, self).__init__()
        self._lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self._attention = PointerAttention(hidden_dim=hidden_size)

    def forward(self, feature, embedds, onehot_mask):
        '''
        feature: (B, emb_dim)
        embedds: (B, max_seq_len, emb_dim) padding
        one-hot mask: (B, max_seq_len+1, num_clusters) eg. [(0, 0, 1, 0, 0, 0, 1, ...), ...]
        '''
        # TODO: the seq_len in a batch is not the same
        # encode the past embeddings
        lstm_embedds, (hn, cn) = self._lstm(embedds)  
        lstm_embedds, out_len = pad_packed_sequence(lstm_embedds, batch_first=True)  # out: (batch, max_seq_len, hidden_size), hn: (batch, num_layers, hidden_size)
        # encode current embeddings
        lstm_cur_embedd, (h_cur, c_cur) = self._lstm(feature, (hn, cn))
        lstm_embedds = torch.cat([lstm_embedds, lstm_cur_embedd], dim=1)
        attention_out = self._attention(lstm_embedds, feature, onehot_mask)  # out: (B, seg_num + 1, num_clusters) or (B, seg_num, num_clusters) if meet the maximum
        q = torch.sum(attention_out, dim=2)
        return q


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

    def forward(self, state):
        embedds = state['embedding_space']  # (B, bin, max_seq_len)
        cur_chunk = state['cur_chunk']  # (B, )
        onehot_mask = state['onehot_mask']
        #centroids = state['centroids']
        
        feature = self._frontend(cur_chunk)
        q = self._backend(feature, embedds, onehot_mask)

        out = {
            'q': q,
            'new_feature': feature
        }

        return out


class obs_buffer:
    '''
    the buffer of state in our case.
    state: {'embedding_space': (embedd_dim, seq_len),
            'cur_chunk': (mel_bin, chunk_len)}
    '''
    def __init__(self, max_size, chunk_len, mel_bin):
        self._embedd_buffer = np.array([None for _ in range(max_size)], object)
        self._chunk_buffer = torch.zeros([max_size, mel_bin, chunk_len])
    
    def add(self, obs, ptr):
        embedds = obs['embedding_space']
        cur_chunk = obs['cur_chunk']
        self._chunk_buffer[ptr] = cur_chunk
        self._embedd_buffer[ptr] = embedds

    def get_batch_by_idx(self, idxs):
        embedds_list = self._embedd_buffer[idxs]
        # sorted by seq_len descending
        embedds_seq_lens = [embedd.size(1) for embedd in embedds_list]
        sorted_idx = np.argsort(-embedds_seq_lens)
        # padding
        embedds_batch = pack_sequence(list(embedds_list[sorted_idx]))
        chunk_batch = self._chunk_buffer[idxs][sorted_idx]
        obs_batch = {
            'embedding_space': embedds_batch,
            'cur_chunk': chunk_batch
        }
        return obs_batch, sorted_idx


class ReplayBuffer:
    def __init__(self, size, n_step=1, priority=True, alpha=0.6, gamma=0.99):
        self._priority = priority
        #self._obs_buffer = torch.zeros([size, obs_shape[0], obs_shape[1]], dtype=torch.float32)
        self._obs_buffer = obs_buffer(size, cfg.CHUNK_LEN, cfg.BIN)
        self._next_obs_buffer = obs_buffer(size, cfg.CHUNK_LEN, cfg.BIN)
        self._action_buffer = torch.zeros([size], dype=torch.int32)
        self._reward_buffer = torch.zeros([size], dtype=torch.float32)
        self._done_buffer = torch.zeros(size, dtype=torch.int32)
        
        self._n_step_buffer = deque(maxlen=n_step)
        self._n_step = n_step

        self._ptr = 0
        self._size = 0
        self._max_size = size
        self._gamma = gamma
    
    def _get_n_step(self):
        """Return n step rew, next_obs, and done."""  
        # info of the last transition
        rew, next_obs, done = self._n_step_buffer[-1][-3:]

        for transition in reversed(list(self._n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + self._gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)  # TODO we will not 

        return rew, next_obs, done

    def store(self, obs, action, reward, next_obs, done):
        transition = (obs, action, reward, next_obs, done)
        self._n_step_buffer.append(transition)
        # when not reaching n_step, return
        if len(self._n_step_buffer) < self._n_step:
            return
        # make a n_step transition
        reward, next_obs, done = self._get_n_step()
        obs, action = self._n_step_buffer[0][:2]

        ptr = self._ptr
        self._obs_buffer.add(obs, ptr)
        self._next_obs_buffer.add(next_obs, ptr)
        self._action_buffer[ptr] = action
        self._reward_buffer[ptr] = reward
        self._done_buffer[ptr] = done
        self._ptr = (ptr+1) % self._max_size
        self._size = min(self._size+1, self._max_size)
        # return 

    def sample(self, batch_size):
        idxs = np.random.choice(self._size, size=batch_size, replace=False)

        obs_batch, sorted_idx = self._obs_buffer.get_batch_by_idx(idxs)
        next_obs_batch, _ = self._next_obs_buffer.get_batch_by_idx(idxs)
        return {
            'obs_batch': obs_batch,
            'next_obs_batch': next_obs_batch,
            'action_batch': self._action_buffer[idxs][sorted_idx],
            'reward_batch': self._reward_buffer[idxs][sorted_idx],
            'done_batch': self._done_buffer[idxs][sorted_idx]
        }
    
    def __len__(self):
        return self._size