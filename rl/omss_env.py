import gym
import numpy as np
import sys

sys.path.append('..')
from utils import config as cfg
import os
from utils.config import eval_hop_size, train_hop_size # TODO: which hop size?
from utils.config import CHUNK_LEN as chunk_len
import torch
import torch.nn.functional as F
import pandas as pd
from . import rl_utils

'''
in environment, the mel and embedding shape is (seq_len, feature_dim)
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
    def __init__(self, frontend_model, num_clusters, file_path, seq_max_len, cluster_encode=True, mode='train'):
        super(OMSSEnv, self).__init__()
        self._mode = mode
        self._num_clusters = num_clusters
        self._frontend_model = frontend_model.eval()  # for updating the embedding space
        self._file_path = file_path  # sample song in order
        self._cluster_encode = cluster_encode

        self._embedding_space = None
        self._seq_max_len = seq_max_len

        self._est_labels = []
        if mode =='train':
            self._hop_size = train_hop_size
        else:
            self._hop_size = eval_hop_size
        
        # reward
        self._times, self._labels = self._load_label()
        self._ref_labels = []
        ## pairwise f1
        self._n_intersection = 0
        self._n_sim_pair_est, self._n_sim_pair_ref = 0, 0
        self._pairwise_f1 = []
    
    def get_max_est_label(self):
        return max(self._est_labels)

    def _load_label(self):
        '''
        load the functions segment labels
        randomly choose 1 or 2 if both are available.
        else, choose the available one
        '''
        # print(self._file_path)
        song_name = self._file_path.split('specs')[-1].split('.')[0][1:-4]
        anno_dir = os.path.join(cfg.SALAMI_DIR, 'annotations', song_name, 'parsed')
        fps = [os.path.join(anno_dir, 'textfile1_functions.txt'),
                os.path.join(anno_dir, 'textfile2_functions.txt')]
        #print(fps)
        if os.path.exists(fps[0]) and os.path.exists(fps[1]):
            fp = fps[int(np.random.rand() > 0.5)]
        elif os.path.exists(fps[0]):
            fp = fps[0]
        elif os.path.exists(fps[1]):
            fp = fps[1]
        else:
            return None, None
        label_df = pd.read_csv(fp, sep='\t', header=None, names=['time', 'label'])
        times, labels = np.array(label_df['time'], dtype='float32'), label_df['label']
        # map labels to numbers
        labels = labels.str.replace('\d+', '', regex=True)
        labels = labels.str.lower()
        labels = labels.str.strip()
        labels = pd.factorize(labels)[0]

        return times, labels
    
    def check_anno(self):
        return self._labels is not None

    def make(self):
        self._mel_spec = np.load(self._file_path)  # pre-calculate mel features
        self._mel_spec = torch.from_numpy(self._mel_spec).transpose(0, 1) # out: (seq_len, feature)
        # ignore first chunk because the label always starts with 0
        self._step_len = ((self._mel_spec.shape[0] - (cfg.CHUNK_LEN - 1) - 1) // self._hop_size)  
        self.reset()
        return self._state
    
    def _update_emb_space(self):
        """
        Decide if we should update embedding space at current step.
        """
        if self._mode == 'infer':
            return False
        return False
    
    def _update_centroids(self, cur_cluster_embedd, cluster_num):
        pre_centroid = self._state['centroids'][cluster_num]
        N = (np.array(self._est_labels) == cluster_num).sum()
        centroid = (N * pre_centroid + cur_cluster_embedd.squeeze(0)) / (N + 1)
        self._state['centroids'][cluster_num] = centroid
    
    def _get_ref_label(self):
        chunk_num = len(self._est_labels)
        start = chunk_num * self._hop_size
        end = start + chunk_len
        label_time = (end - cfg.time_lag_len) * cfg.BIN_TIME_LEN
        label_idx = np.argmax((self._times > label_time)) - 1
        
        return self._labels[label_idx]


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
        # ref labels
        ref_label = self._get_ref_label()
        self._ref_labels.append(ref_label)
        reward = self._get_reward(cur_state, action)

        # if reaching the end of a song, return done
        if self._cur_chunk_idx + self._hop_size + chunk_len >= self._mel_spec.shape[0]:
            done = True
            reward += self._pairwise_f1[-1]  # get a reward on the end of the episode
            return self.reset(), reward, done, info
        
        # 2. update state
        ### a. update clusters
        cluster_num = int(action['action'])
        new_feature = action['new_feature']  # (1, feature)
        cur_cluster_embedd = action['cur_cluster_embedd']
        self._est_labels.append(cluster_num)
        # cluster encode
        cluster_encode = F.one_hot(torch.tensor([cluster_num]), num_classes=self._num_clusters)  # out: (1, num_classes)
        if self._cluster_encode:
            new_feature = torch.cat([new_feature, cluster_encode], dim=1)
        
        ### b. concatenate new feature to embedding space (seq_len, feature)
        self._embedding_space = torch.cat([self._state['embedding_space'], new_feature], dim=0)
        if self._embedding_space.shape[0] > self._seq_max_len:
            self._state['embedding_space'] = self._embedding_space[-self._seq_max_len:]
        else:
            self._state['embedding_space'] = self._embedding_space
        
        # c. update the centroids using ... TODO
        self._update_centroids(cur_cluster_embedd, cluster_num)

        # ### c. update onehot_mask for attention (num_clusters, seq_len)
        # onehot_mask = self._state['onehot_mask'] 
        
        # est_labels = self._est_labels
        # if est_labels[-1] == est_labels[-2]: # if same clusters, zero the last one hot
        #     onehot_mask[-1, :] = torch.zeros(self._num_clusters)
        # onehot_mask = torch.cat([onehot_mask, cluster_encode], dim=0)
        # self._state['onehot_mask'] = onehot_mask

        ### d. update embedding space
        if self._update_emb_space():
            # update embedding space using frontend model
            pass

        ### e. new coming chunk
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
        # print(self._cur_chunk_idx)
        return self._state, reward, done, info

    def _get_reward(self, state, action):
        cluster_num = int(action['action'])
        new_feature = action['new_feature']  # (1, feature)

        # pairwise f1
        sim_pair_vt_est = (np.array(self._est_labels) == cluster_num)
        n_sim_pair_est = sim_pair_vt_est.sum()

        sim_pair_vt_ref = (np.array(self._ref_labels[:-1]) == self._ref_labels[-1])
        n_sim_pair_ref = sim_pair_vt_ref.sum()

        matches = np.logical_and(sim_pair_vt_est, sim_pair_vt_ref)
        n_matches = matches.sum()

        self._n_sim_pair_est += n_sim_pair_est
        self._n_sim_pair_ref += n_sim_pair_ref
        self._n_intersection += n_matches

        # print(self._est_labels)
        # print(self._ref_labels)
        # print(n_matches)
        # print(sim_pair_vt_est)
        # print()
        # prc = n_matches / n_sim_pair_est if n_sim_pair_est != 0 else 0
        # rec = n_matches / n_sim_pair_ref if n_sim_pair_ref != 0 else 0
        prc = self._n_intersection / self._n_sim_pair_est if self._n_sim_pair_ref != 0 else 0
        rec = self._n_intersection / self._n_sim_pair_ref if self._n_sim_pair_ref != 0 else 0
        f1 = rl_utils.f1_measure(prc, rec, beta=0.5)  # beta=0.5 when more weights on prec
        step = len(self._est_labels)
        #print(step/self._step_len)
        f1 = f1 * step / self._step_len
        #print(f1)
        self._pairwise_f1.append(f1)
        f1_diff = self._pairwise_f1[-1] - self._pairwise_f1[-2]
        # if cluster_num != self._est_labels[-1] and self._ref_labels[-2] != self._ref_labels[-1]:
        #     boundary_reward = 1
        # else:
        #     boundary_reward = 0
        # return torch.tensor(f1 + boundary_reward)
        return torch.tensor(f1_diff)

    def reset(self):
        """
        Reset the environment.
        1. embedd the first chunk
        2. move the chunk idx pointer
        """
        # initialize labels
        self._est_labels.append(0)
        self._ref_labels.append(0)

        self._pairwise_f1.append(0)
        # embedd the first chunk
        chunk = self._mel_spec[0:chunk_len, :].to(next(self._frontend_model.parameters()).device)
        #print(chunk.shape)
        self._frontend_model.eval()
        with torch.no_grad():
            embedd = self._frontend_model(chunk.unsqueeze(0)).detach().cpu()  # (1, feature)
        #print(embedd.shape)
        self._frontend_model.train()
        self._cur_chunk_idx = self._hop_size
        cluster_encode = F.one_hot(torch.tensor([0]), num_classes=self._num_clusters)
        if self._cluster_encode:
            embedd = torch.cat([embedd, cluster_encode], dim=1)
        #onehot_mask = cluster_encode
        centroids = torch.zeros([self._num_clusters, embedd.size(1)])
        centroids[0, :] = embedd
        self._embedding_space = embedd
        self._state = {'embedding_space': embedd, 
                        'cur_chunk': self._mel_spec[self._hop_size:self._hop_size+chunk_len, :],
                        'centroids': centroids}
        return self._state

    def seed(self, seed):
        np.random.seed(seed)