from utils import config as cfg
from rl import rl_model, omss_env, rl_utils
import torch
import argparse
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(args):
    model_path = '/home/zewen/online-music-structure-segmentation/rl/experiments/04111452/best_q_net.pth'
    song_path = '/storage/zewen/salami-data-public/melspecs/338-mel.npy'
    backend_input_size = cfg.EMBEDDING_DIM + args.num_clusters if args.cluster_encode else cfg.EMBEDDING_DIM
    nets = [rl_model.QNet(input_shape=(cfg.BIN, cfg.CHUNK_LEN),
                            embedding_size=backend_input_size,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            num_heads=args.num_heads,
                            num_clusters=args.num_clusters,
                            cluster_encode=args.cluster_encode,
                            freeze_frontend=args.freeze_frontend,
                            use_rnn=args.use_rnn,
                            mode='test').to(device) for _ in range(2)]
    q_net = nets[0]
    # load pretrained model
    if len(args.pretrained) > 3:
        q_net.load_frontend(args.pretrained)
        print('load pretrained frontend model!')
    
    q_net.load_state_dict(torch.load(model_path)['state_dict'])
    print('load best q net!')
    # set target network to eval mode
    target_q_net = nets[1]
    target_q_net.load_state_dict(q_net.state_dict())
    target_q_net.eval()

    # policy
    policy = rl_utils.Policy(q_net=q_net, 
                            target_q_net=target_q_net, 
                            gamma=args.gamma,
                            target_update_freq=args.target_update_freq,
                            n_step=args.n_step,
                            optim=None,
                            device=device)
    
    env = omss_env.OMSSEnv(#q_net.module.get_frontend(), 
                                    q_net.get_frontend(),
                                    args.num_clusters, 
                                    song_path, 
                                    args.seq_max_len,  # TODO don't need this in val
                                    cluster_encode=args.cluster_encode, 
                                    mode='test')

    done = False
    score = 0
    state = env.make()
    done = False
    while not done:
        # print(state)
        action = policy.take_action(state, env, args.test_eps, args.num_clusters)
        next_state, reward, done, info = env.step(action)
        state = next_state
        score += reward.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    so_far_best_pretrained = os.path.join(cfg.SUP_EXP_DIR, '03020414', 'unsup_embedding_best.pt')

    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--test_idxs', type=str, default=None)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    # embedding model
    parser.add_argument('--pretrained', type=str, default=so_far_best_pretrained)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_size', type=int, default=128)  #*
    parser.add_argument('--num_layers', type=int, default=1)    #*
    parser.add_argument('--num_heads', type=int, default=1)   # *
    # rl
    # backend
    parser.add_argument('--seq_max_len', type=int, default=128)
    parser.add_argument('--num_clusters', type=int, default=5)  # *
    parser.add_argument('--cluster_encode', action='store_true')
    #parser.add_argument('--max_grad_norm', type=float, default=2.0)
    parser.add_argument('--freeze_frontend', action='store_true')

    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_update_freq', type=int, default=100)
    ## eps greedy search
    parser.add_argument('--train_eps', type=float, default=1.)  
    parser.add_argument('--final_train_eps', type=float, default=0.05)
    parser.add_argument('--train_eps_decay', type=float, default=1/2000)
    parser.add_argument('--test_eps', type=float, default=0.)
    ## priority buffer
    parser.add_argument('--priority', action='store_true')
    parser.add_argument('--prior_eps', type=float, default=1e-6)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta_anneal_step', type=int, default=1e5)  # 400 * n_song
    parser.add_argument('--final_beta', type=float, default=1.)
    ## buffer
    parser.add_argument('--buffer_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    ## n step
    parser.add_argument('--n_step', type=int, default=3)
    args = parser.parse_args()
    test(args)