from utils import config as cfg
import torch
import argparse
import os
from tianshou_rl_train import get_args, omss_train_val_test_split, state_to
from rl import tianshou_rl_model, tianshou_env
from tianshou.policy import DQNPolicy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(args):
    # prepare dataset (file paths)
    test_idxs = None
    if args.test_idxs:
        # load test set indexs TODO
        pass
    else:
        train_dataset, val_dataset, test_dataset = omss_train_val_test_split(cfg.val_pct, cfg.test_pct, test_idxs, args)
    
    # define model
    backend_input_size = cfg.EMBEDDING_DIM + args.num_clusters if args.cluster_encode else cfg.EMBEDDING_DIM
    net = tianshou_rl_model.QNet(
                            input_shape=(cfg.BIN, cfg.CHUNK_LEN),
                            embedding_size=backend_input_size,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            num_heads=args.num_heads,
                            num_clusters=args.num_clusters,
                            cluster_encode=args.cluster_encode,
                            use_rnn=args.use_rnn,
                            device=device,
                            freeze_frontend=args.freeze_frontend)

    # policy
    # define policy
    policy = DQNPolicy(
        model=net,
        optim=None,
        discount_factor=args.gamma,
        target_update_freq=args.target_update_freq,
        is_double=True
    ).to(device)

    # load pretrained model
    checkpoint = torch.load(args.resume_path, map_location=device)
    policy.load_state_dict(checkpoint['state_dict'])
    best_score = checkpoint['best_score']
    print('best score: ', best_score)
    print("Loaded agent from: ", args.resume_path)

    # result dir
    res_dir = os.path.join(args.resume_path.split('.')[0][:-12], 'test_figs')
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    print(res_dir)
    # test loop
    q_net = policy.model
    q_net.eval()
    frontend = q_net.get_frontend()
    score = 0
    f1 = 0
    with torch.no_grad():
        with trange(len(val_dataset)) as t:
            for k in t:
                fp = val_dataset[k]
                print(fp)
                env = tianshou_env.OMSSEnv(#q_net.module.get_frontend(), 
                                        frontend,
                                        args.num_clusters, 
                                        fp, 
                                        args.seq_max_len,  # TODO don't need this in val
                                        cluster_encode=args.cluster_encode, 
                                        mode='test')
                song_action_list = [0]
                state = env.reset()
                done = False
                while not done:
                    format_state = state_to(state, device)
                    logits = policy.model(format_state)[0].detach().cpu().numpy()
                    # print(logits)
                    action = np.argmax(logits)
                    # print(action)
                    song_action_list.append(int(action))
                    
                    # action = policy.take_action(state, env, args.test_eps, args.num_clusters)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    score += reward
                f1 += info['f1']
                t.set_description('f1: {}'.format(info['f1']))

                # plot result for current song
                x = (np.arange(len(song_action_list)) * cfg.eval_hop_size * cfg.BIN_TIME_LEN \
                    + (cfg.CHUNK_LEN - cfg.time_lag_len) * cfg.BIN_TIME_LEN)
                plt.figure(figsize=(12.8, 4.8))
                plt.plot(x, song_action_list, 'o', markersize=2)
                plt.xlabel('time / s')
                plt.yticks(range(0, args.num_clusters))
                plt.ylabel('action')
                song_num = fp.split('specs')[-1].split('.')[0][1:-4]
                plt.title('Song No. {}, f1: {}'.format(song_num, info['f1']))
                #print(res_dir)
                plt.savefig(os.path.join(res_dir, song_num + '.jpg'))
                plt.close()

        score /= len(val_dataset)
        f1 /= len(val_dataset)

if __name__ == '__main__':
    args = get_args()
    test(args)