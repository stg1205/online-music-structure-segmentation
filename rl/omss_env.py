import gym
import numpy as np
import sys

from sklearn import cluster
sys.path.append('..')
from utils import config as cfg
import os
from utils.config import eval_hop_size, train_hop_size # TODO: which hop size?
from utils.config import CHUNK_LEN as chunk_len
import librosa
import torch
import torch.nn.functional as F
from copy import deepcopy

'''
in environment, the mel and embedding shape is (feature_dim, seq_len)
'''
class OMSSEnv(gym.Env):
    """
    Description:
        The RL environment of online music structure segmentation.

    Observation:
        Type: np.ndarray
        The embedding space of the past chunks and new coming chunk

    Actions:
        Type: Discrete(num_clusters)
        assign the new chunk to the one of the cluster

    Reward: 
        1. clustering based reward:
        2. segment based reward: 

    Episode Termination:
        When a song ends.
    """
    def __init__(self, frontend_model, num_clusters, file_path, cluster_encode=True, mode='train'):
        super(OMSSEnv, self).__init__()
        self._mode = mode
        self._num_clusters = num_clusters
        self._frontend_model = frontend_model.eval()  # for updating the embedding space
        self._file_path = file_path  # sample song in order
        self._cluster_encode = cluster_encode

        self._clusters = []
        if mode =='train':
            self._hop_size = train_hop_size
        else:
            self._hop_size = eval_hop_size

    def make(self, device):
        self._mel_spec = np.load(self._file_path)  # pre-calculate mel features
        self._mel_spec = torch.from_numpy(self._mel_spec).transpose(0, 1) # out: (seq_len, feature)
        self.reset(device)
        return self._state
    
    def _update_emb_space(self):
        """
        Decide if we should update embedding space at current step.
        """
        if self._mode == 'infer':
            return False
        return False
    
    def _update_centroids(self, new_feature, cluster_num):
        pass

    def step(self, action):
        """
        Accepts an action and returns a tuple (observation, reward, done, info).
        Receive the action taken by the agent.
        Update the state by 
        1. add the new coming chunk to state
        2. update the cluster
        2. update the embedding space using feature front-end model if meet the requirements

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        done = False
        info = {}
        # 1. calculate reward
        cur_state = self._state
        reward = self._get_reward(cur_state, action)

        # if reaching the end of a song, return done
        if self._cur_chunk_idx + self._hop_size + chunk_len >= self._mel_spec.shape[0]:
            done = True
            return None, reward, done, info
        
        # 2. update state
        ### a. update clusters
        cluster_num = int(action['action'])
        new_feature = action['new_feature']  # (1, feature)
        
        cluster_encode = F.one_hot(torch.tensor([cluster_num]), num_classes=self._num_clusters)  # out: (1, num_classes)
        if self._cluster_encode:
            new_feature = torch.cat([new_feature, cluster_encode], dim=1)
        
        ### concatenate new feature to embedding space (seq_len, feature)
        self._state['embedding_space'] = torch.cat([self._state['embedding_space'], new_feature], dim=0)

        # update the centroids using ... TODO
        # self._state['centroids'] = self._update_centroids(new_feature, cluster_num)

        ### update onehot_mask for attention (num_clusters, seq_len)
        onehot_mask = self._state['onehot_mask']
        self._clusters.append(cluster_num)
        labels = self._clusters
        if labels[-1] == labels[-2]: # if same clusters, zero the last one hot
            onehot_mask[-1, :] = torch.zeros(self._num_clusters)
        onehot_mask = torch.cat([onehot_mask, cluster_encode], dim=0)
        self._state['onehot_mask'] = onehot_mask

        ### b. update embedding space
        if self._update_emb_space():
            # update embedding space using frontend model
            pass

        ### c. new coming chunk
        start = self._cur_chunk_idx + self._hop_size
        self._cur_chunk_idx = start
        end = start + chunk_len
        self._state['cur_chunk'] = self._mel_spec[start:end, :]

        # onehot mask encode the next cluster number if not reaching the maximum cluster number
        # after that, just cat zeros to aviod new clusters, do attention to the previous embeddings
        # TODO can't encode here because padding
        # if max(labels) + 1 < self._num_clusters:
        #     next_mask = F.one_hot(max(labels)+1, num_classes=self._num_clusters).unsqueeze(0)
        # else:
        #     next_mask = torch.zeros((1, self._num_clusters))
        # onehot_mask = torch.cat([onehot_mask, next_mask], dim=0)
        
        return self._state, reward, done, info
        
    def _get_reward(self, state, aciton):
        return torch.tensor(1)

    def reset(self, device):
        """
        Reset the environment.
        1. embedd the first chunk
        2. move the chunk idx pointer
        """
        chunk = self._mel_spec[0:chunk_len, :].to(device)
        self._clusters.append(0)
        #print(chunk.shape)
        self._frontend_model.eval()
        embedd = self._frontend_model(chunk.unsqueeze(0)).detach().cpu()  # (1, feature)
        #print(embedd.shape)
        self._frontend_model.train()
        self._cur_chunk_idx = self._hop_size
        cluster_encode = F.one_hot(torch.tensor([0]), num_classes=self._num_clusters)
        if self._cluster_encode:
            embedd = torch.cat([embedd, cluster_encode], dim=1)
        onehot_mask = cluster_encode
        self._state = {'embedding_space': embedd, 
                        'cur_chunk': self._mel_spec[self._hop_size:self._hop_size+chunk_len, :],
                        'onehot_mask': onehot_mask}
        return self._state

    def seed(self, seed):
        np.random.seed(seed)