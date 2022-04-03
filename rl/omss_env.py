import gym
import numpy as np
import sys

from sklearn import cluster
sys.path.append('..')
from utils import config as cfg
import os
from utils.config import eval_hop_size as hop_size # TODO: which hop size?
from utils.config import CHUNK_LEN as chunk_len
import librosa
import torch
import torch.nn.functional as F

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
    def __init__(self, frontend_model, num_clusters, cluster_encoder, file_path, mode='train'):
        super(self, OMSSEnv).__init__()
        self._mode = mode
        self._num_clusters = num_clusters
        self._frontend_model = frontend_model.eval()  # for updating the embedding space
        self._file_list = self._load_dataset()
        self._file_path = file_path  # sample song in order
        self._cluster_encoder = cluster_encoder

    def make(self):
        self._mel_spec = np.load(self._file_path)  # pre-calculate mel features
        self._mel_spec = torch.from_numpy(self._mel_spec)
        self.reset()
        return self._state

    def _load_dataset(self):
        """
        read file list in SALAMI dataset
        """
        files = os.listdir(cfg.SALAMI_DIR)
        # np.random.shuffle(files)
        return list(map(lambda x: os.path.join(cfg.SALAMI_DIR, x), files))
    
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
        if self._cur_chunk_idx + hop_size + chunk_len >= self._mel_spec.shape[1]:
            done = True
            return None, reward, done, info
        
        # 2. update state
        ### a. update clusters
        q = action['q']
        new_feature = action['new_feature']
        cluster_num = torch.argmax(q)
        if self._cluster_encoder:
            cluster_encode = F.one_hot(cluster_num, num_classes=self._num_clusters)
            new_feature = torch.cat([new_feature, cluster_encode])
        
        ### concatenate new feature to embedding space
        if self._state['embedding_space'] != None:
            self._state['embedding_space'] = torch.cat([self._state['embedding_space'], new_feature.unsqueeze(1)], dim=1)
        else:
            self._state['embedding_space'] = new_feature.squeeze(1)

        # update the centroids using ... TODO
        # self._state['centroids'] = self._update_centroids(new_feature, cluster_num)

        ### update onehot_mask for attention
        self._state['clusters'].append(int(cluster_num))
        labels = self._state['clusters']
        if len(labels) == 1:
            onehot_mask = cluster_encode.unsqueeze(1)
        elif labels[-1] == labels[-2]:
            onehot_mask = torch.cat([onehot_mask, cluster_encode.unsqueeze(1)], dim=1)
            onehot_mask[:, -2] = torch.zeros(self._num_clusters)
        else:
            onehot_mask = torch.cat([onehot_mask, cluster_encode.unsqueeze(1)], dim=1)

        ### b. update embedding space
        if self._update_emb_space():
            # update embedding space using frontend model
            pass

        ### c. new coming chunk
        start = self._cur_chunk_idx + hop_size
        self._cur_chunk_idx = start
        end = start + chunk_len
        self._state['cur_chunk'] = self._mel_spec[:, start:end]
        return self._state, reward, done, info
        
    def _get_reward(self, state, aciton):
        pass

    def reset(self):
        """
        Reset the environment by moving to next song. 
        """
        # update file index
        #self._file_idx = self._file_idx + 1 if self._file_idx < len(self._file_list) else 0
        
        self._cur_chunk_idx = 0
        self._state = {'embedding_space': None, 
                        'clusters': [],
                        'centroids': np.zeros((cfg.EMBEDDING_DIM, self._num_clusters)),
                        'cur_chunk': self._mel_spec[:, 0:chunk_len],
                        'onehot_mask': None}
        return self._state

    def seed(self, seed):
        np.random.seed(seed)