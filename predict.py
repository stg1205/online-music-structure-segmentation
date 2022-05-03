from utils.modules import split_to_chunk_with_hop
from utils import config as cfg
import torch
from supervised_model.sup_model import Frontend
from tianshou.policy import DQNPolicy
from rl import tianshou_rl_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_embedding(model_name, model_path, song):
    '''
    Predict embeddings given the pretrained model and the spectrogram of a song
    '''
    # preprocessing
    song = torch.tensor(song)
    n_chunks = ((song.shape[1] - (cfg.CHUNK_LEN - 1) - 1) // cfg.train_hop_size) + 1
    data_len = cfg.CHUNK_LEN + (n_chunks - 1) * cfg.train_hop_size
    song = song[:, :data_len]

    # initialize model
    if model_name == 'unsup_embedding':
        model = Frontend((cfg.BIN, cfg.CHUNK_LEN), embedding_dim=32)
    elif model_name == 'rl':
        backend_input_size = cfg.EMBEDDING_DIM + 5
        net = tianshou_rl_model.QNet(
                            input_shape=(cfg.BIN, cfg.CHUNK_LEN),
                            embedding_size=backend_input_size,
                            hidden_size=128,
                            num_layers=1,
                            num_heads=1,
                            num_clusters=5,
                            cluster_encode=True,
                            use_rnn=True,
                            device=device,
                            freeze_frontend=False)

        # policy
        # define policy
        model = DQNPolicy(
            model=net,
            optim=None,
            discount_factor=0.99,
            target_update_freq=500,
            is_double=True
        )
    
    # load model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])

    if model_name == 'rl':
        model = model.model._frontend
    
    model.eval()
    
    song = split_to_chunk_with_hop(song, cfg.eval_hop_size)
    embeddings = model(song)

    song_embedding = torch.transpose(embeddings, 0, 1)

    return song_embedding