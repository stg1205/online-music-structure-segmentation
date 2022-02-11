import pandas as pd
import numpy as np
import torch
import torch.utils.data as tud
import os
import config
import scipy


class HarmonixDataset(tud.Dataset):
    '''Harmonix dataset object.

    __getitem__(self, index) will return:
    1. the spectrogram of a song
    2. the segment label (an integer) for each chunk. 
        ex. np.array([0, 0, 0,..., 1, 1, 1,..., 0, 0, 0...])
    
    To follow the idea of the original paper, each chunk is aligned with beat or downbeat. However,
    in our online model we may not get the beat information. Thus, we use a simple overlapped 
    sample method. The length of the chunk is defined in config.py. It also contains a hop size 
    in case we need.
    '''
    def __init__(self, data='melspecs', 
                        data_format='npy',
                        transform=None,
                        alignment=None):
        self._data_paths = []
        self._label_paths = []
        
        # load the data paths and label paths to memory
        data_dir = os.path.join(config.DATASET, data)
        label_dir = os.path.join(config.DATASET, 'segments')
        for f in os.listdir(data_dir):
            if not f.endswith(data_format):
                continue
            data_path = os.path.join(data_dir, f)
            song_name = f[:-8]  # -mel.npy
            label_path = os.path.join(label_dir, song_name+'.txt')

            self._data_paths.append(data_path)
            self._label_paths.append(label_path)
        
        self._transform = transform
        self._alignment = alignment

    def __len__(self):
        return len(self._data_paths)
    
    def __getitem__(self, index):
        data = np.load(self._data_paths[index])  # (mels, time)
        print(self._data_paths[index])
        if self._transform is not None:
            data = self._transform(data)
        
        label = pd.read_csv(self._label_paths[index], sep=' ', header=None, names=['time', 'label'])
        
        times, labels = np.array(label['time'], dtype='float32'), label['label']
        # map labels to numbers
        label_dict = {}
        labels = labels.str.replace('\d+', '')
        labels = labels.str.lower()
        labels = labels.str.strip()
        for i, l in enumerate(labels.drop_duplicates()):
            label_dict[l] = i
        print(label_dict)
        if not self._alignment:
            hop_size = config.HOP_SIZE
            chunk_len = config.CHUNK_LEN
            bin_time_len = config.BIN_TIME_LEN

            # TODO: double check this number of chunks and data length for a song
            n_chunks = ((data.shape[1] - (chunk_len - 1) - 1) // hop_size) + 1
            data_len = chunk_len + (n_chunks - 1) * hop_size
            data = data[:, :data_len]
            
            # map each chunk to segment label
            chunk_labels = []
            for i in range(n_chunks):
                start = i * hop_size * bin_time_len
                end = start + chunk_len * bin_time_len
                center = (start + end) / 2

                # the label is represented by center of this chunk
                label_idx = np.argmax(times > center) - 1
                #print(label_idx)
                if label_idx < 0: # some silence may not be labeled like before starting or after ends
                    chunk_labels.append(-1)
                else:
                    chunk_labels.append(label_dict[labels[label_idx]])
        else:
            raise NotImplementedError

        return data, np.array(chunk_labels)


class SongDataset(tud.Dataset):
    '''Batch sampling within a song
    
    '''
    def __init__(self, data, labels, transform=None, alignment=None):
        self._data = data
        self._labels = labels
        self._alignment = alignment
        self._transform = transform  # TODO: window function
    
    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, index):
        if not self._alignment:
            start = config.HOP_SIZE * index
            end = start + config.CHUNK_LEN

            return self._data[start:end, :], self._labels[index]
        else:
            raise NotImplementedError



if __name__ == '__main__':
    dataset = HarmonixDataset()

    dataloader = tud.DataLoader(dataset, 1, shuffle=True, num_workers=0)

    sample, label = next(iter(dataloader))
    print(sample, sample.shape)
    print(label, label.shape)