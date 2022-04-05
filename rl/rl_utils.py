import sys
sys.path.append('..')
from utils import config as cfg
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import numpy as np
import torch
from collections import deque
from .segment_tree import MinSegmentTree, SumSegmentTree
import random


class obs_buffer:
    '''
    the buffer of state in our case.
    state: {'embedding_space': (embedd_dim, seq_len),
            'cur_chunk': (mel_bin, chunk_len)}
    '''
    def __init__(self, max_size, chunk_len, mel_bin):
        self._embedd_buffer = np.array([None for _ in range(max_size)], object)
        self._chunk_buffer = torch.zeros([max_size, chunk_len, mel_bin])
        self._onehot_mask_buffer = np.array([None for _ in range(max_size)], object)
    
    def add(self, obs, ptr):
        embedds = obs['embedding_space']
        cur_chunk = obs['cur_chunk']
        onehot_mask = obs['onehot_mask']
        self._chunk_buffer[ptr] = cur_chunk.squeeze(0)
        self._embedd_buffer[ptr] = embedds.squeeze(0)
        self._onehot_mask_buffer[ptr] = onehot_mask.squeeze(0)

    def get_batch_by_idx(self, idxs):
        embedds_list = self._embedd_buffer[idxs]
        mask_list = self._onehot_mask_buffer[idxs]
        # sorted by seq_len descending
        embedds_seq_lens = np.array([embedd.size(0) for embedd in embedds_list])
        sorted_idx = np.argsort(-embedds_seq_lens)
        # padding
        embedds_batch = pack_padded_sequence(
            pad_sequence(list(embedds_list[sorted_idx]), batch_first=True), 
            lengths=torch.tensor(embedds_seq_lens[sorted_idx]), batch_first=True)
        mask_batch = pad_sequence(list(mask_list[sorted_idx]), batch_first=True)
        
        # batch states
        chunk_batch = self._chunk_buffer[idxs][sorted_idx]
        obs_batch = {
            'embedding_space': embedds_batch,
            'cur_chunk': chunk_batch,
            'onehot_mask': mask_batch
        }
        return obs_batch, sorted_idx


class ReplayBuffer:
    def __init__(self, size, n_step=1, priority=True, alpha=0.6, gamma=0.99):
        self._priority = priority
        #self._obs_buffer = torch.zeros([size, obs_shape[0], obs_shape[1]], dtype=torch.float32)
        self._obs_buffer = obs_buffer(size, cfg.CHUNK_LEN, cfg.BIN)
        self._next_obs_buffer = obs_buffer(size, cfg.CHUNK_LEN, cfg.BIN)
        self._action_buffer = torch.zeros(size, dtype=torch.int64)
        self._reward_buffer = torch.zeros(size, dtype=torch.float32)
        self._done_buffer = torch.zeros(size, dtype=torch.int64)
        
        self._n_step_buffer = deque(maxlen=n_step)
        self._n_step = n_step

        self._ptr = 0
        self._size = 0
        self._max_size = size
        self._gamma = gamma

        # priority buffer
        if self._priority:
            self.max_priority, self.tree_ptr = 1.0, 0
            self.alpha = alpha
            
            # capacity must be positive and a power of 2.
            tree_capacity = 1
            while tree_capacity < self._max_size:
                tree_capacity *= 2
            
            self.tree_capacity = tree_capacity
            self.sum_tree = SumSegmentTree(tree_capacity)
            self.min_tree = MinSegmentTree(tree_capacity)
    
    def reset(self):
        self._size = 0
        self._ptr = 0
        self.tree_ptr = 0
        self._n_step_buffer = deque(maxlen=self._n_step)
        self.sum_tree = SumSegmentTree(self.tree_capacity)
        self.min_tree = MinSegmentTree(self.tree_capacity)

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
        
        # priority buffer
        if self._priority:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self._max_size
        return self._n_step_buffer[0]

    def sample(self, batch_size, beta):
        if self._priority:
            idxs = self._sample_proportional(batch_size)
        else:
            idxs = np.random.choice(self._size, size=batch_size, replace=False)

        return self.sample_from_idxs(idxs, beta)
    
    def sample_from_idxs(self, idxs, beta):
        obs_batch, sorted_idx = self._obs_buffer.get_batch_by_idx(idxs)
        next_obs_batch, _ = self._next_obs_buffer.get_batch_by_idx(idxs)
        if self._priority:
            weights = torch.tensor([self._calculate_weight(i, beta) for i in idxs])
            return {
                'obs_batch': obs_batch,
                'next_obs_batch': next_obs_batch,
                'action_batch': self._action_buffer[idxs][sorted_idx],
                'reward_batch': self._reward_buffer[idxs][sorted_idx],
                'done_batch': self._done_buffer[idxs][sorted_idx],
                'weights': weights,
                'idxs': idxs
            }
        else:
            return {
                'obs_batch': obs_batch,
                'next_obs_batch': next_obs_batch,
                'action_batch': self._action_buffer[idxs][sorted_idx],
                'reward_batch': self._reward_buffer[idxs][sorted_idx],
                'done_batch': self._done_buffer[idxs][sorted_idx],
                'idxs': idxs
            }
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self, batch_size):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight
    
    def __len__(self):
        return self._size
