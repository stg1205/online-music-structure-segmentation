from math import ceil
import pandas as pd
import numpy as np
import torch
import torch.utils.data as tud
import os
import sys
sys.path.append('..')
from utils import config as cfg


class HarmonixDataset(tud.Dataset):
    '''Harmonix dataset object.

    To follow the idea of the original paper, each chunk is aligned with beat or downbeat. However,
    in our online model we may not get the beat information. Thus, we use a simple overlapped 
    sample method. The length of the chunk is defined in config.py. It also contains a hop size 
    in case we need.
    '''
    def __init__(self, data='melspecs', 
                        data_format='npy',
                        transform=None):
        self._data_paths = []
        self._label_paths = []
        
        # load the data paths and label paths to memory
        data_dir = os.path.join(cfg.DATASET, data)
        label_dir = os.path.join(cfg.DATASET, 'segments')
        for f in os.listdir(data_dir):
            if not f.endswith(data_format):
                continue
            data_path = os.path.join(data_dir, f)
            song_name = f[:-8]  # -mel.npy
            label_path = os.path.join(label_dir, song_name+'.txt')

            self._data_paths.append(data_path)
            self._label_paths.append(label_path)
        
        self._transform = transform

    def __len__(self):
        return len(self._data_paths)
    
    def __getitem__(self, index):
        data = np.load(self._data_paths[index])  # (mels, time)
        #print(self._data_paths[index])
        if self._transform is not None:
            data = self._transform(data)
        
        label_df = pd.read_csv(self._label_paths[index], sep=' ', header=None, names=['time', 'label'])
        
        times, labels = np.array(label_df['time'], dtype='float32'), label_df['label']
        # map labels to numbers
        labels = labels.str.replace('\d+', '')
        labels = labels.str.lower()
        labels = labels.str.strip()
        labels = pd.factorize(labels)[0]

        # return ref_times and ref_labels for msaf algorithms
        return {'data': torch.tensor(data),
                'ref_times': times,
                'ref_labels': np.array(labels)}


class SongDataset(tud.Dataset):
    '''Batch sampling within a song

    __getitem__(self, index) will return:
    1. the ith chunk of a song
    2. the segment label (an integer) for the chunk. 
    
    `alignment`: if align the chunks with beat
    '''
    def __init__(self, data, times, labels, batch_size, mode,
                transform=None, alignment=None, label_strategy='last'):
        self._data = data
        self._times = times
        self._labels = labels
        self._mode = mode
        self._alignment = alignment
        self._transform = transform  # TODO: window function
        self._batch_size = batch_size
        self._label_strategy = label_strategy

        self._chunks, self._chunk_labels = self._process_data()
    
    def _process_data(self):
        '''
        1. split the data to chunks with hop size for train and validation dataset respectively
        2. calculate the label of each chunk
        3. shuffle the samples
        4. complement the last batch
        '''
        # different hop size for train and validation set
        if self._mode == 'train':
            hop_size = cfg.train_hop_size
        elif self._mode == 'val':
            hop_size = cfg.eval_hop_size
        chunk_len = cfg.CHUNK_LEN
        bin_time_len = cfg.BIN_TIME_LEN
        
        # split chunks and map them to labels
        if not self._alignment:
            # TODO: double check this number of chunks and data length for a song
            n_chunks = ((self._data.shape[1] - (chunk_len - 1) - 1) // hop_size) + 1
            data_len = chunk_len + (n_chunks - 1) * hop_size
            self._data = self._data[:, :data_len]
            
            # map each chunk to a segment label
            chunks = []
            chunk_labels = []
            for i in range(n_chunks):
                start = i * hop_size
                end = start + chunk_len
                chunks.append(self._data[:, start:end])

                # decide which time the label is represented by
                if self._label_strategy == 'center':
                    label_time = (start + end) * bin_time_len / 2
                elif self._label_strategy == 'last':  # TODO: some window function should be applied to the chunk to emphasize current frame
                    label_time = (end - cfg.time_lag_len) * bin_time_len  # about 1s time lag

                label_idx = torch.argmax((self._times > label_time).type(torch.uint8)) - 1
                #print(label_idx)
                if label_idx < 0: # some silence may not be labeled, like before starting or after ends
                    chunk_labels.append(-1)
                else:
                    chunk_labels.append(self._labels[label_idx])
        else:
            raise NotImplementedError
        
        # we don't have to shuffle and complement for the validation set
        chunks = torch.stack(chunks)
        chunk_labels = torch.tensor(chunk_labels)
        if self._mode == 'val':
            return chunks, chunk_labels
        
        # shuffle
        rands = torch.randperm(chunk_labels.size(0))
        chunks = chunks[rands, :, :]
        chunk_labels = chunk_labels[rands]
        
        # complement the last batch
        num_batch = chunk_labels.size(0) / self._batch_size
        if num_batch > 1:
            r = ceil(num_batch)*self._batch_size - len(chunk_labels)
            chunks = torch.concat([chunks, chunks[:r, :, :]])
            chunk_labels = torch.concat([chunk_labels, chunk_labels[:r]])
        return chunks, chunk_labels

    def __len__(self):
        return len(self._chunk_labels)
    
    def __getitem__(self, index):
        return self._chunks[index, :, :], self._chunk_labels[index]


if __name__ == '__main__':
    dataset = HarmonixDataset()

    dataloader = tud.DataLoader(dataset, 1, shuffle=True, num_workers=0)

    batch = next(iter(dataloader))
    print(batch)

    song_dataset = SongDataset(batch['data'].squeeze(0), 
                            batch['ref_times'].squeeze(0), 
                            batch['ref_labels'].squeeze(0), 128, 'train')
    song_dataloader = tud.DataLoader(song_dataset, 1)
    song_batch = next(iter(song_dataloader))
    print(song_batch)
    