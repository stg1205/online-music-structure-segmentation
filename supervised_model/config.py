DATASET = 'C:\\UR\\AIR Lab\\thesis\\dataset\\harmonixset-master\\dataset'

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
1 / 22050 * 1024
'''
BIN_TIME_LEN = 1 / 22050 * 1024

CHUNK_LEN = 32
HOP_SIZE = 16