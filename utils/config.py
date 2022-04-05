import os
# colab
HARMONIX_DIR = '/content/drive/MyDrive/thesis/dataset/harmonixset-master/dataset'
WORK_DIR = '/content/drive/Othercomputers/MyEnvy/online-music-structure-segmentation'
# local
# DATASET = 'C:\\UR\AIR Lab\\thesis\\dataset\\harmonixset-master\\dataset'
WORK_DIR = 'C:\\UR\\AIR Lab\\thesis\\online-music-structure-segmentation'
SALAMI_DIR = 'C:\\UR\\AIR Lab\\thesis\\dataset\\SALAMI\\melspecs'
SUP_DIR = WORK_DIR + '\\supervised_model\\experiments\\'
RL_DIR = WORK_DIR + '\\rl\\experiments\\'


'''
{
    "librosa_version": "0.7.0",
    "numpy_version": "1.17.2",
    "SR": 22050,
    "N_MELS": 80,
    "N_FFT": 2048,
    "HOP_LENGTH": 1024,
    "MEL_FMIN": 0,
    "MEL_FMAX": null
}
1 / 22050 * 1024 = 0.046s
'''

# feature parameters
sr = 22050
n_mel = 80
n_fft = 2048
mel_hop = 1024
mel_fmin = 0
mel_fmax = None

BIN_TIME_LEN = 1 / 22050 * 1024
CHUNK_LEN = 64
BIN = 80
EMBEDDING_DIM = 128

train_hop_size = 10
eval_hop_size = 2
time_lag_len = 20

# Spectral Clustering Params
scluster_config = {
    "num_layers" : 10,   # How many hierarchical layers to compute (only for the hierarchical case)
    "scluster_k" : 4,    # How many unique labels to have (only for the flat case)
    "evec_smooth": 5,
    "rec_smooth" : 3,
    "rec_width"  : 2,
    "hier": False
}

val_pct = 0.1
test_pct = 0.1