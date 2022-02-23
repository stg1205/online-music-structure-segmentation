from utils.modules import split_to_chunk_with_hop
from utils import config as cfg
import torch
from supervised_model.sup_model import UnsupEmbedding

def predict_embedding(model_name, model_path, song):
    '''
    Predict embeddings given the pretrained model and the spectrogram of a song
    '''
    # preprocessing
    song = torch.tensor(song)
    n_chunks = ((song.shape[1] - (cfg.CHUNK_LEN - 1) - 1) // cfg.HOP_SIZE) + 1
    data_len = cfg.CHUNK_LEN + (n_chunks - 1) * cfg.HOP_SIZE
    song = song[:, :data_len]

    # initialize model
    if model_name == 'unsup_embedding':
        model = UnsupEmbedding((cfg.BIN, cfg.CHUNK_LEN))
    
    # load model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    song = split_to_chunk_with_hop(song)
    embeddings = model(song)

    song_embedding = torch.transpose(embeddings, 0, 1)

    return song_embedding