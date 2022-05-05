from tracemalloc import start
import gym
from gym import spaces
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
from scipy.optimize import linear_sum_assignment


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
    def __init__(self, 
                frontend_model, 
                num_clusters, 
                file_path, 
                seq_max_len, 
                freeze_frontend,
                final_eps=0.05,
                final_punish=-5,
                eps=1.0, 
                cluster_encode=True, 
                mode='train'):
        super(OMSSEnv, self).__init__()
        self._mode = mode
        self._freeze_frontend = freeze_frontend
        self._num_clusters = num_clusters
        self._frontend_model = frontend_model.eval()  # for updating the embedding space
        self._file_path = file_path  # sample song in order
        self._cluster_encode = cluster_encode

        self._embedding_space = None
        self._seq_max_len = seq_max_len

        # reward function is related to exploration
        self._eps = eps  
        self._final_punish = final_punish
        self._final_eps = final_eps
        
        if mode =='train':
            self._hop_size = train_hop_size
        else:
            self._hop_size = eval_hop_size
        
        # reward
        self._times, self._labels = self._load_label()
        
        # tianshou
        self.action_space = gym.spaces.Discrete(self._num_clusters)
        embedd_size = cfg.EMBEDDING_DIM + num_clusters if cluster_encode else cfg.EMBEDDING_DIM

        if not freeze_frontend:
            self.observation_space = spaces.Dict({
                'embedding_space': spaces.Box(low=np.Inf, high=np.Inf, shape=(seq_max_len, embedd_size)),
                'cur_chunk': spaces.Box(low=np.Inf, high=np.Inf, shape=(cfg.CHUNK_LEN, cfg.BIN)),
                'centroids': spaces.Box(low=np.Inf, high=np.Inf, shape=(num_clusters, embedd_size)),
                #'lens': spaces.Discrete(128, start=1)
                'lens': spaces.Box(low=1, high=seq_max_len+1, shape=(1, 1))
                })
        else:
            self.observation_space = spaces.Dict({
                'embedding_space': spaces.Box(low=np.Inf, high=np.Inf, shape=(seq_max_len, embedd_size)),
                'cur_embedding': spaces.Box(low=np.Inf, high=np.Inf, shape=(1, embedd_size)),
                'centroids': spaces.Box(low=np.Inf, high=np.Inf, shape=(num_clusters, embedd_size)),
                #'lens': spaces.Discrete(128, start=1)
                'lens': spaces.Box(low=1, high=seq_max_len+1, shape=(1, 1))
                })
        # print(self._file_path)
        self._mel_spec = np.load(self._file_path).T  # pre-calculate mel features
        # self._mel_spec = torch.from_numpy(self._mel_spec).transpose(0, 1) # out: (seq_len, feature)
        # ignore first chunk because the label always starts with 0
        self._step_len = ((self._mel_spec.shape[0] - (cfg.CHUNK_LEN - 1) - 1) // self._hop_size)  
        self.reset()
    
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
        if cfg.dataset == 'salami':
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
        elif cfg.dataset == 'harmonix':
            fp = os.path.join(cfg.HARMONIX_DIR, 'segments', song_name+'.txt')
            label_df = pd.read_csv(fp, sep=' ', header=None, names=['time', 'label'])
        
        times, labels = np.array(label_df['time'], dtype='float32'), label_df['label']
        # map labels to numbers
        labels = labels.str.replace(r'\d+', '', regex=True)
        labels = labels.str.lower()
        labels = labels.str.strip()
        labels = pd.factorize(labels)[0]

        return times, labels
    
    def check_anno(self):
        return self._labels is not None
    
    def _update_emb_space(self):
        """
        Decide if we should update embedding space at current step.
        """
        if self._mode == 'infer':
            return False
        return False
    
    def _update_centroids(self, cur_cluster_embedd, cluster_num):
        pre_centroid = self._state['centroids'][cluster_num]
        # average embeddings to update centroid
        # N = (np.array(self._est_labels) == cluster_num).sum() - 1
        # centroid = (N * pre_centroid + cur_cluster_embedd.squeeze(0)) / (N + 1)

        # exp moving average
        gamma = 0.567
        centroid = (1-gamma) * pre_centroid + gamma * cur_cluster_embedd
        self._state['centroids'][cluster_num] = centroid
    
    def _get_ref_label(self):
        chunk_num = len(self._est_labels)
        start = chunk_num * self._hop_size
        end = start + chunk_len
        label_time = (end - cfg.time_lag_len) * cfg.BIN_TIME_LEN
        label_idx = np.argmax((self._times > label_time)) - 1
        
        # print(chunk_num, label_time)
        return self._labels[label_idx]

    def _embedd_chunk(self, chunk):
        chunk = torch.as_tensor(chunk)
        self._frontend_model.eval()
        with torch.no_grad():
            new_feature = self._frontend_model(chunk.unsqueeze(0).transpose(-1, -2))
        
        return new_feature.numpy()

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

        #print(self._cur_chunk_idx)
        # if reaching the end of a song, return done
        if self._cur_chunk_idx + self._hop_size + chunk_len >= self._mel_spec.shape[0]:
            done = True
            info['f1'] = self._pairwise_f1[-1] 
            # if info['f1'] >= 60:
            #     reward += 100  # get a reward on the end of the episode
            return self.reset(), reward, done, info
        
        # 2. update state
        ### a. update clusters
        if not self._freeze_frontend:
            new_feature = self._embedd_chunk(self._state['cur_chunk'])
        else:
            new_feature = self._state['cur_embedding']

        cluster_num = action
        self._est_labels.append(cluster_num)
        # cluster encode
        if self._cluster_encode:
            cluster_encode = np.zeros([1, self._num_clusters])  # out: (1, num_classes)
            cluster_encode[:, cluster_num] = 1
            new_feature = np.concatenate([new_feature, cluster_encode], axis=1)
        
        ### b. concatenate new feature to embedding space (seq_len, feature)
        self._embedding_space = np.concatenate([self._embedding_space, new_feature], axis=0)
        if self._embedding_space.shape[0] > self._seq_max_len:
            self._state['embedding_space'] = self._embedding_space[-self._seq_max_len:]
            self._state['lens'] =  np.array([[self._seq_max_len]])
        else:
            zero_padds = np.zeros(
                [self._seq_max_len - self._embedding_space.shape[0], new_feature.shape[1]])
            self._state['embedding_space'] = np.concatenate([self._embedding_space, zero_padds], axis=0)
            self._state['lens'] = np.array([[self._embedding_space.shape[0]]])
        
        # c. update the centroids using moving average
        cur_cluster_embedd = new_feature
        self._update_centroids(cur_cluster_embedd, cluster_num)

        ### d. update embedding space
        if self._update_emb_space():
            # update embedding space using frontend model
            pass

        ### e. new coming chunk
        start = self._cur_chunk_idx + self._hop_size
        self._cur_chunk_idx = start
        end = start + chunk_len

        if not self._freeze_frontend:
            self._state['cur_chunk'] = self._mel_spec[start:end, :]
        else:
            self._state['cur_embedding'] = self._embedd_chunk(self._mel_spec[start:end, :])

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

    def _cal_pairwise_f1(self, est_label):
        # pairwise f1
        sim_pair_vt_est = (np.array(self._est_labels) == est_label)
        n_sim_pair_est = sim_pair_vt_est.sum()

        sim_pair_vt_ref = (np.array(self._ref_labels[:-1]) == self._ref_labels[-1])
        n_sim_pair_ref = sim_pair_vt_ref.sum()

        matches = np.logical_and(sim_pair_vt_est, sim_pair_vt_ref)
        n_matches = matches.sum()

        self._n_sim_pair_est += n_sim_pair_est
        self._n_sim_pair_ref += n_sim_pair_ref
        self._n_intersection += n_matches

        prc = self._n_intersection / self._n_sim_pair_est if self._n_sim_pair_est != 0 else 0
        rec = self._n_intersection / self._n_sim_pair_ref if self._n_sim_pair_ref != 0 else 0
        f1 = rl_utils.f1_measure(prc, rec, beta=1.0)  # beta=0.5 when more weights on prec
        step = len(self._est_labels)
        #print(step/self._step_len)
        f1 = f1 * step / self._step_len * 100
        self._pairwise_f1.append(f1)

    def _bipartite_match(self, graph):
        """Find maximum cardinality matching of a bipartite graph (U,V,E).
        The input format is a dictionary mapping members of U to a list
        of their neighbors in V.
        The output is a dict M mapping members of V to their matches in U.
        Parameters
        ----------
        graph : dictionary : left-vertex -> list of right vertices
            The input bipartite graph.  Each edge need only be specified once.
        Returns
        -------
        matching : dictionary : right-vertex -> left vertex
            A maximal bipartite matching.
        """
        # Adapted from:
        #
        # Hopcroft-Karp bipartite max-cardinality matching and max independent set
        # David Eppstein, UC Irvine, 27 Apr 2002

        # initialize greedy matching (redundant, but faster than full search)
        matching = {}
        for u in graph:
            for v in graph[u]:
                if v not in matching:
                    matching[v] = u
                    break

        while True:
            # structure residual graph into layers
            # pred[u] gives the neighbor in the previous layer for u in U
            # preds[v] gives a list of neighbors in the previous layer for v in V
            # unmatched gives a list of unmatched vertices in final layer of V,
            # and is also used as a flag value for pred[u] when u is in the first
            # layer
            preds = {}
            unmatched = []
            pred = dict([(u, unmatched) for u in graph])
            for v in matching:
                del pred[matching[v]]
            layer = list(pred)

            # repeatedly extend layering structure by another pair of layers
            while layer and not unmatched:
                new_layer = {}
                for u in layer:
                    for v in graph[u]:
                        if v not in preds:
                            new_layer.setdefault(v, []).append(u)
                layer = []
                for v in new_layer:
                    preds[v] = new_layer[v]
                    if v in matching:
                        layer.append(matching[v])
                        pred[matching[v]] = v
                    else:
                        unmatched.append(v)

            # did we finish layering without finding any alternating paths?
            if not unmatched:
                unlayered = {}
                for u in graph:
                    for v in graph[u]:
                        if v not in preds:
                            unlayered[v] = None
                return matching

            def recurse(v):
                """Recursively search backward through layers to find alternating
                paths.  recursion returns true if found path, false otherwise
                """
                if v in preds:
                    L = preds[v]
                    del preds[v]
                    for u in L:
                        if u in pred:
                            pu = pred[u]
                            del pred[u]
                            if pu is unmatched or recurse(pu):
                                matching[v] = u
                                return True
                return False

            for v in unmatched:
                recurse(v)

    def _get_reward(self, state, action):
        '''
        reward function
        '''
        est_label = action

        # pairwise f1 for current chunk
        self._cal_pairwise_f1(est_label)
        f1_diff = self._pairwise_f1[-1] - self._pairwise_f1[-2]

        # if skip clustering, like [0, 0, 0, 2], get -1 punishment
        # if action - np.max(self._est_labels) > 1:
        #     skip_cluster_punish = -1
        
        # reward = skip_cluster_punish + f1_diff  # scale cumulated reward to [0, 1]

        # boundary reward for testing how the agent will behave to the reward
        # if cluster_num != self._est_labels[-1] and self._ref_labels[-2] != self._ref_labels[-1]:
        #     boundary_reward = 1
        # else:
        #     boundary_reward = 0
        #reward = boundary_reward

        # bipartitie graph matching and compare the mapped label with gt
        ## construct graph by step
        self._step += 1
        ref_label = self._ref_labels[-1]

        # weighted
        self._cost_mat[est_label, ref_label] -= 1
        # print(self._cost_mat)
        est_idx, ref_idx = linear_sum_assignment(self._cost_mat)
        # print(est_idx, ref_idx, est_label)
        reward = 1 if ref_idx[est_idx == est_label] == ref_label else 0

        # # not weighted
        # G = self._G
        # if ref_label not in G:
        #     G[ref_label] = set()
        # G[ref_label].add(est_label)
        
        # ## update graph matching when reach the freq
        # # if self._graph_step - self._pre_graph_update_step == cfg.graph_update_freq or \
        # #     est_label not in self._matching:
        # if self._step - self._pre_graph_update_step == cfg.graph_update_freq \
        #     or self._ref_labels[-1] != self._ref_labels[-2] \
        #     or len(self._est_labels) == 1:
        #     self._matching = self._bipartite_match(G)
        #     self._pre_graph_update_step = self._step
        
        # mapped_est_label = self._matching.get(est_label, -1)
        # reward = 1 if mapped_est_label == self._ref_labels[-1] else 0

        # huge punishment if not meet the segment length requirement
        if self._est_labels[-1] != est_label:
            # measure the length of last segment
            if self._step - self._last_boundary_step < cfg.MIN_SEG_LEN / cfg.BIN_TIME_LEN / self._hop_size:
                # filter the correct boundary predictions
                if self._ref_labels[-1] == self._ref_labels[-2]:
                    punish = self._final_punish / self._final_eps * self._eps
                    print(punish)
                    reward += punish
            # else:
            #     reward += 1
            
            self._last_boundary_step = self._step


        # print(self._file_path)
        # print('est_labels: ', self._est_labels + [est_label])
        # print('ref_labels: ', self._ref_labels)
        # print('graph: ', G)
        # print('matching: ', self._matching)
        return reward

    def _init_centroids(self, embedd_dim):
        centroids = np.random.rand(self._num_clusters, embedd_dim)
        centroids = centroids / np.linalg.norm(centroids, ord=2, axis=1, keepdims=True)

        return centroids

    def reset(self):
        """
        Reset the environment.
        1. embedd the first chunk
        2. move the chunk idx pointer
        """
        # initialize labels
        self._est_labels = []
        ref_label = self._get_ref_label()
        self._ref_labels = [ref_label]
        self._est_labels.append(0)

        ## pairwise f1
        self._n_intersection = 0
        self._n_sim_pair_est, self._n_sim_pair_ref = 0, 0
        self._pairwise_f1 = [0] # f1 measure is 0 at beginning

        self._step = 0
        # bipartie graph matching
        # the first est label is 0
        # self._G = {}  # ref: est
        # self._G[ref_label] = set()
        # self._G[ref_label].add(0)
        # self._matching = {ref_label: 0}
        # self._pre_graph_update_step = 0
        # self._last_boundary_step = 0
        
        # weighted graph (using number of matching labels as weights)
        self._cost_mat = np.zeros([self._num_clusters, len(set(self._labels))])
        self._cost_mat[0, ref_label] -= 1

        # embedd the first chunk
        embedd = self._embedd_chunk(self._mel_spec[0:chunk_len, :])
        # print(embedd)
        self._cur_chunk_idx = self._hop_size

        # cluster encode
        cluster_encode = np.zeros([1, self._num_clusters])  # out: (1, num_classes)
        cluster_encode[:, 0] = 1
        if self._cluster_encode:
            embedd = np.concatenate([embedd, cluster_encode], axis=1)
        
        # centroids
        centroids = self._init_centroids(embedd.shape[1])
        gamma = 0.567
        centroids[0] = (1-gamma) * centroids[0] + gamma * embedd

        # zero padd
        zero_padds = np.zeros([self._seq_max_len-1, embedd.shape[1]])
        self._embedding_space = embedd

        if not self._freeze_frontend:
            self._state = {'embedding_space': np.concatenate([embedd, zero_padds], axis=0), 
                            'cur_chunk': self._mel_spec[self._hop_size:self._hop_size+chunk_len, :],
                            'lens': np.array([[1]]),
                            'centroids': centroids}
        else:
            cur_embedding = self._embedd_chunk(self._mel_spec[self._hop_size:self._hop_size+chunk_len, :])
            self._state = {'embedding_space': np.concatenate([embedd, zero_padds], axis=0), 
                            'cur_embedding': cur_embedding,
                            'lens': np.array([[1]]),
                            'centroids': centroids}
        # print(self._state['cur_chunk'])
        return self._state

    def seed(self, seed):
        np.random.seed(seed)
    
    def render(self, mode):
        pass

    def close(self):
        pass
