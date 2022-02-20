DATASET = '/content/drive/MyDrive/thesis/dataset/harmonixset-master/dataset'
WORK_DIR = '/content/drive/Othercomputers/MyEnvy/online-music-structure-segmentation'

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
BIN_TIME_LEN = 1 / 22050 * 1024

CHUNK_LEN = 64
HOP_SIZE = 32
BIN = 80