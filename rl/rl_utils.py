import sys
sys.path.append('..')
from utils import config as cfg
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import numpy as np
import torch
from collections import deque
from .segment_tree import MinSegmentTree, SumSegmentTree
import random
import wandb


class obs_buffer:
    '''
    the buffer of state in our case.
    state: {'embedding_space': (embedd_dim, seq_len),
            'cur_chunk': (mel_bin, chunk_len)}
    '''
    def __init__(self, max_size, chunk_len, mel_bin, num_clusters, hidden_size):
        self._embedd_buffer = np.array([None for _ in range(max_size)], object)
        self._chunk_buffer = torch.zeros([max_size, chunk_len, mel_bin])
        self._centroids_buffer = torch.zeros([max_size, num_clusters, hidden_size])
    
    def add(self, obs, ptr):
        embedds = obs['embedding_space']
        cur_chunk = obs['cur_chunk']
        centroids = obs['centroids']
        self._chunk_buffer[ptr] = cur_chunk
        self._embedd_buffer[ptr] = embedds
        self._centroids_buffer[ptr] = centroids

    def get_batch_by_idx(self, idxs):
        embedds_list = self._embedd_buffer[idxs]
        # sorted by seq_len descending
        embedds_seq_lens = torch.tensor([embedd.size(0) for embedd in embedds_list])
        # sorted_idx = np.argsort(-embedds_seq_lens)
        # padding
        # embedds_batch = pack_padded_sequence(
        #     pad_sequence(list(embedds_list[sorted_idx]), batch_first=True), 
        #     lengths=torch.tensor(embedds_seq_lens[sorted_idx]), batch_first=True)
        # print(embedds_seq_lens)
        embedds_batch = pad_sequence(list(embedds_list), batch_first=True)
        # batch states
        # chunk_batch = self._chunk_buffer[idxs][sorted_idx]
        # centroids_batch = self._centroids_buffer[idxs][sorted_idx]
        chunk_batch = self._chunk_buffer[idxs]
        centroids_batch = self._centroids_buffer[idxs]
        obs_batch = {
            'embedding_space': embedds_batch,
            'cur_chunk': chunk_batch,
            'centroids': centroids_batch
        }
        return obs_batch, embedds_seq_lens


class ReplayBuffer:
    def __init__(self, size, num_clusters, hidden_size, n_step=1, priority=True, alpha=0.6, gamma=0.99):
        self._priority = priority
        #self._obs_buffer = torch.zeros([size, obs_shape[0], obs_shape[1]], dtype=torch.float32)
        self._obs_buffer = obs_buffer(size, cfg.CHUNK_LEN, cfg.BIN, num_clusters, hidden_size)
        self._next_obs_buffer = obs_buffer(size, cfg.CHUNK_LEN, cfg.BIN, num_clusters, hidden_size)
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
        #print(idxs)
        return self.sample_from_idxs(idxs, beta)
    
    def sample_from_idxs(self, idxs, beta):
        obs_batch, obs_seq_lens = self._obs_buffer.get_batch_by_idx(idxs)
        next_obs_batch, next_obs_seq_lens = self._next_obs_buffer.get_batch_by_idx(idxs)
        if self._priority:
            weights = torch.tensor(np.array([self._calculate_weight(i, beta) for i in idxs]))
        else:
            weights = torch.ones(len(idxs))
        return {
            'obs_batch': obs_batch,
            'obs_seq_lens': obs_seq_lens,
            'next_obs_batch': next_obs_batch,
            'next_obs_seq_lens': next_obs_seq_lens,
            'action_batch': self._action_buffer[idxs],
            'reward_batch': self._reward_buffer[idxs],
            'done_batch': self._done_buffer[idxs],
            'weights': weights,
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


def state_to(state, device):
    # add a batch dimension and move to device
    if len(state['embedding_space'].size()) < 3:
        embedds = state['embedding_space'].unsqueeze(0).to(device)
        cur_chunk = state['cur_chunk'].unsqueeze(0).transpose(-1, -2).to(device)
        centroids = state['centroids'].unsqueeze(0).to(device)
    else:
        embedds = state['embedding_space'].to(device)
        cur_chunk = state['cur_chunk'].transpose(-1, -2).to(device)
        centroids = state['centroids'].to(device)
    return embedds, cur_chunk, centroids

def state_out(out):
    out['q'] = out['q'].detach().cpu()
    out['new_feature'] = out['new_feature'].detach().cpu()
    out['cur_cluster_embedd'] = out['cur_cluster_embedd'].detach().cpu()


class Policy:
    def __init__(self, 
                q_net, 
                target_q_net, 
                gamma,
                target_update_freq,
                n_step,
                optim,
                device):
        self._q_net = q_net
        self._target_q_net = target_q_net
        # params
        self._gamma = gamma
        self._target_update_freq = target_update_freq
        self._n_step = n_step

        self._update_count = 0

        self._optim = optim
        self._device = device

    def take_action(self, state, env, eps, num_clusters):
        '''
        select action using epsilon greedy policy
        only can choose up to cur cluster number + 1
        '''
        #print('take action')
        self._q_net.eval()
        # predict q value
        format_state = state_to(state, self._device)  # has to contain a batch dimension
        #self._q_net.module.set_mode('infer')
        self._q_net.set_mode('infer')
        with torch.no_grad():
            out = self._q_net(*format_state)
        #self._q_net.module.set_mode('train')
        self._q_net.set_mode('train')
        self._q_net.train()
        state_out(out)
        feature = out['new_feature']
        cur_cluster_embedd = out['cur_cluster_embedd']
        try:
            wandb.log({'explore/q': torch.argmax(out['q'])})
        except:
            pass
        #print(out['q'])
        #print(torch.argmax(out['q']))

        if eps > np.random.random():
            cur_max_cluster = env.get_max_est_label()
            if cur_max_cluster + 1 < num_clusters:
                upbound = cur_max_cluster + 2
            else:
                upbound = num_clusters
            selected_action = torch.randint(0, upbound, (1,))
        else:
            selected_action = torch.argmax(out['q'])
        action = {
            'action': selected_action,
            'cur_cluster_embedd': cur_cluster_embedd,
            'new_feature': feature
        }
        return action
    
    def _dqn_loss(self, batch, gamma):
        '''
        calculate double dqn loss
        '''
        state = state_to(batch["obs_batch"], self._device)
        state_lens = batch['obs_seq_lens']
        next_state = state_to(batch["next_obs_batch"], self._device)
        next_state_lens = batch['next_obs_seq_lens']
        action = batch["action_batch"].reshape(-1, 1).to(self._device)
        reward = batch["reward_batch"].reshape(-1, 1).to(self._device)
        done = batch["done_batch"].reshape(-1, 1).to(self._device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
            #       = r                       otherwise
        cur_q_value = self._q_net(*state, state_lens)['q'].gather(1, action)
        next_q_value = self._target_q_net(*next_state, next_state_lens)['q'].gather(  # Double DQN
                1, self._q_net(*next_state, next_state_lens)['q'].argmax(dim=1, keepdim=True)
            ).detach()
        mask = 1 - done
        target = (reward + gamma * next_q_value * mask).to(self._device)

        # try:
        #     q_log = {
        #         'train/cur_q': torch.mean(cur_q_value),
        #         'train/next_q': torch.mean(next_q_value),
        #         'train/target_q': torch.mean(target)
        #     }
        #     wandb.log(q_log)
        # except:
        #     pass

        #print(state)
        # calculate dqn loss
        loss = F.smooth_l1_loss(cur_q_value, target, reduction='none')
        
        return loss
    
    def update(self, batch, n_batch, optim):
        # calculate loss by combining one step and n step loss
        ele_wise_loss = self._dqn_loss(batch, self._gamma)
        if self._n_step > 1:
            gamma = self._gamma ** self._n_step
            ele_wise_n_loss = self._dqn_loss(n_batch, gamma)
            ele_wise_loss += ele_wise_n_loss
        
        weights = batch['weights'].to(self._device)
        loss = torch.mean(ele_wise_loss * weights)  # TODO: which weights?
        # back propagation
        optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self._q_net._backend.parameters(), 0.5)
        self._optim.step()

        # hard update target network
        self._update_count += 1
        if self._update_count % self._target_update_freq == 0:
            self._target_q_net.load_state_dict(self._q_net.state_dict())
        
        return ele_wise_loss, loss


def f1_measure(prc, rec, beta=1.0):
    if prc == 0 or rec == 0:
        return 0
    
    return (1 + beta**2) * prc * rec / ((beta**2) * prc + rec)
