import gym
import numpy as np
from . import config as cfg
import os


class MSSEnv(gym.Env):
    """
    Description:
        The RL environment of online music structure segmentation.

    Observation:
        Type: np.ndarray
        The embedding space of the past chunks and new coming chunk

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Not merge
        1     merge the new chunk to the closest cluster

    Reward:
        1. clustering based reward:
        2. segment based reward: 

    Starting State:
        The embedding space is empty.

    Episode Termination:
        When a song ends.
    """
    def __init__(self, frontend_model,
                        reward_model, 
                        batch_size=1) -> None:
        super().__init__()
        self._file_list = self._load_dataset()
        self._file_idx = 0
        self._batch_start_idx = 0
        self._batch_size = batch_size  # sample how many songs without clearing the embedding space
        
        self._cur_mel_spec = np.load(self._file_list[0])
        self._cur_chunk_idx = cfg.HOP_SIZE

        self._state = {'embedding_space': np.array([]), 
                        'clusters': np.array([]),
                        'cur_chunk': self._cur_mel_spec[0:cfg.CHUNK_LEN]}
        self._frontend_model = frontend_model
        self._reward_model = reward_model

    def _load_dataset(self):
        """read file list in dataset directory
        """
        files = os.listdir(cfg.DATASET_DIR)
        np.random.shuffle(files)
        return list(map(lambda x: os.path.join(cfg.DATASET_DIR, x), files))

    def step(self, action):
        """Accepts an action and returns a tuple (observation, reward, done, info).
        Receive the action taken by the agent.
        Update the state by 
        1. add the new coming chunk to state
        2. update the cluster
        2. update the embedding space using feature front-end model

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

        # 2. update state
        #### a. new coming chunk
        if self._cur_chunk_idx + HOP_SIZE + CHUNK_LEN < self._cur_mel_spec.shape[0]:  # TODO: mel shape
            start = self._cur_chunk_idx + HOP_SIZE
            self._cur_chunk_idx = start
            end = start + CHUNK_LEN
            self._state['cur_chunk'] = self._cur_mel_spec[start:end]
        else:
            # have to sample a new song
            self._file_idx += 1
            if self._file_idx == len(self._file_list):
                # episode ends
                done = True
                self.reset()
                return self._state, reward, done, info
            
            if self._file_idx - self._batch_start_idx == self._batch_size:
                # if reach the batch size, clear the embedding space and clusters
                self._state['embedding_space'] = np.array([])
                self._state['clusters'] = np.array([])
                self._batch_start_idx = self._file_idx

            # move to next song
            self._cur_mel_spec = np.load(self._file_list[self._file_idx])
            cur_chunk = self._cur_mel_spec[:, 0:CHUNK_LEN]
            self._state['cur_chunk'] = cur_chunk
            self._cur_chunk_idx = HOP_SIZE
        
        #### b. update clusters
        
        #### c. embedding space

        
        return self._state, reward, done, info
        
    def _get_reward(self, state, action):
        pass

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        self._file_idx = 0
        self._batch_start_idx = 0
        np.random.shuffle(self._file_list)
        self._cur_mel_spec = np.load(self._file_list[0])
        self._cur_chunk_idx = HOP_SIZE

        self._state = {'embedding_space': np.array([]), 
                        'clusters': np.array([]),
                        'cur_chunk': self._cur_mel_spec[:, 0:CHUNK_LEN]}
        return self._state

    def render(self, mode="human"):
        pass

    def seed(self, seed): # TODO
        np.random.seed(seed)
