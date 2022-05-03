from supervised_model.sup_model import Frontend
from utils import config as cfg
import torch
import argparse
import os
from tianshou_rl_train import get_args, omss_train_val_test_split, state_to, omss_train_val_split
from rl import tianshou_rl_model, tianshou_env
from tianshou.policy import DQNPolicy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from links_cluster import LinksCluster
from utils.msaf_validation import eval_seg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # prepare dataset (file paths)
    test_files = None
    if cfg.test_csv:
        # load test set indexs TODO
        import pandas as pd
        val_files = np.array(pd.read_csv(cfg.test_csv, header=None)[0])
        train_dataset, val_dataset = omss_train_val_split(cfg.val_pct, val_files, args)
    else:
        train_dataset, val_dataset, test_dataset = omss_train_val_test_split(cfg.val_pct, cfg.test_pct, test_files, args)
    
    # define model
    backend_input_size = cfg.EMBEDDING_DIM + args.num_clusters if args.cluster_encode else cfg.EMBEDDING_DIM
    if not args.freeze_frontend:
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
                                freeze_frontend=args.freeze_frontend
        )
        if args.pretrained:
            net.load_frontend(args.pretrained)
    else:
        net = tianshou_rl_model.TianshouBackend(input_size=backend_input_size,
                                        hidden_size=args.hidden_size,
                                        num_layers=args.num_layers,
                                        num_clusters=args.num_clusters,
                                        num_heads=args.num_heads,
                                        mode='train',
                                        use_rnn=args.use_rnn,
                                        device=device,
                                        cluster_encode=args.cluster_encode)
        
        checkpoint = torch.load(args.pretrained)
        frontend = Frontend((cfg.BIN, cfg.CHUNK_LEN), embedding_dim=cfg.EMBEDDING_DIM)
        frontend.load_state_dict(checkpoint['state_dict'])

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
    exp_dir = args.resume_path.split('/best')[0]
    res_dir = os.path.join(exp_dir, args.resume_path.split('/best_')[1].split('_policy')[0])
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    print(res_dir)
    # test loop
    q_net = policy.model
    q_net.eval()
    if not args.freeze_frontend:
        frontend = q_net.get_frontend()
    
    metrics = {
        'HitRate_3F': 0, 
        'HitRate_3P': 0,
        'HitRate_3R': 0,
        'HitRate_0.5F': 0, 
        'HitRate_0.5P': 0,
        'HitRate_0.5R': 0,
        'PWF': 0, 
        'PWP': 0,
        'PWR': 0,
        'score': 0
    }
    with torch.no_grad():
        with trange(len(val_dataset)) as t:
            for k in t:
                fp = val_dataset[k]
                # if not fp.split('/')[-1].startswith('0603'):
                #     continue
                print(fp)
                env = tianshou_env.OMSSEnv(#q_net.module.get_frontend(), 
                                        frontend,
                                        args.num_clusters, 
                                        fp, 
                                        args.seq_max_len,  # TODO don't need this in val
                                        cluster_encode=args.cluster_encode, 
                                        freeze_frontend=args.freeze_frontend,
                                        mode='test')
                song_action_list = [0]
                state = env.reset()
                label_list = [env._ref_labels[0]]
                done = False
                song_score = 0
                song_reward_list = [0]
                est_idxs = [0]
                pre_action = 0
                idx = 1
                while not done:
                    format_state = state_to(state, device, args=args)
                    logits = policy.model(format_state)[0].detach().cpu().numpy()
                    action = np.argmax(logits)
                    song_action_list.append(int(action))
                    label_list.append(env._ref_labels[-1])

                    next_state, reward, done, info = env.step(action)
                    state = next_state

                    metrics['score'] += reward
                    song_score += reward
                    song_reward_list.append(int(reward))

                    # boundary evaluation
                    if int(action) != pre_action:
                        pre_action = action
                        est_idxs.append(idx)
                    
                    idx += 1
                
                # plot result for current song
                times = (np.arange(len(song_action_list)) * cfg.eval_hop_size * cfg.BIN_TIME_LEN \
                    + (cfg.CHUNK_LEN - cfg.time_lag_len) * cfg.BIN_TIME_LEN)
                
                plt.rcParams['figure.figsize'] = (20, 12)
                plt.subplot(3, 1, 1)
                plt.plot(times, song_action_list, 'o', markersize=2)
                plt.xlabel('time / s')
                plt.yticks(range(0, args.num_clusters))
                plt.ylabel('action')
                song_num = fp.split('specs')[-1].split('.')[0][1:-4]
                plt.title('Song No. {}, f1: {}, score: {}'.format(song_num, info['f1'], song_score))

                plt.subplot(3, 1, 2)
                plt.plot(times, song_reward_list, 'o', markersize=2)
                plt.xlabel('time / s')
                plt.ylabel('reward')

                plt.subplot(3, 1, 3)
                plt.plot(times, label_list, 'o', markersize=2)
                plt.xlabel('time / s')
                plt.ylabel('label')
                #print(res_dir)
                plt.savefig(os.path.join(res_dir, song_num + '.jpg'))
                plt.close()

                # boundary evaluation              
                est_seg_labels = np.array(song_action_list)[est_idxs]
                est_idxs.append(len(song_action_list) - 1)
                
                est_times = times[est_idxs]
                
                res = eval_seg(est_times, 
                                est_seg_labels, 
                                env._times, 
                                env._labels[:-1])
                
                for k in metrics:
                    if k != 'score':
                        metrics[k] += res[k]

                t.set_description('f1: {}, '.format(info['f1']))
                with open(os.path.join(res_dir, song_num + '.json'), 'w') as f:
                    f.write(str(res).replace("'", '"'))
                    f.close()

        # save overall result
        for k in metrics:
            metrics[k] /= len(val_dataset)
        with open(os.path.join(res_dir, 'res.json'), 'w') as f:
            f.write(str(metrics).replace("'", '"'))
            f.close()


def test_links(args):
    # prepare dataset (file paths)
    test_files = None
    if cfg.test_csv:
        # load test set indexs TODO
        pass
    else:
        train_dataset, val_dataset, test_dataset = omss_train_val_test_split(cfg.val_pct, cfg.test_pct, test_files, args)

    checkpoint = torch.load(args.pretrained)
    frontend = Frontend((cfg.BIN, cfg.CHUNK_LEN), embedding_dim=cfg.EMBEDDING_DIM)
    frontend.load_state_dict(checkpoint['state_dict'])

    # result dir
    exp_dir = 'links'
    res_dir = os.path.join(exp_dir, cfg.dataset)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    print(res_dir)
    
    agent = LinksCluster(cluster_similarity_threshold=0.9, 
                subcluster_similarity_threshold=0.8, 
                pair_similarity_maximum=1.0, 
                store_vectors=True)
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
                                        freeze_frontend=args.freeze_frontend,
                                        mode='test')
                song_action_list = [0]
                state = env.reset()
                label_list = [env._ref_labels[0]]
                done = False
                song_score = 0
                song_reward_list = [0]
                while not done:
                    action = agent.predict((state['cur_embedding']))

                    # print(action)
                    song_action_list.append(int(action))
                    label_list.append(env._ref_labels[-1])

                    # action = policy.take_action(state, env, args.test_eps, args.num_clusters)
                    next_state, reward, done, info = env.step(action)
                    state = next_state
                    score += reward
                    song_score += reward
                    song_reward_list.append(int(reward))
                f1 += info['f1']
                t.set_description('f1: {}'.format(info['f1']))

                # plot result for current song
                x = (np.arange(len(song_action_list)) * cfg.eval_hop_size * cfg.BIN_TIME_LEN \
                    + (cfg.CHUNK_LEN - cfg.time_lag_len) * cfg.BIN_TIME_LEN)
                plt.rcParams['figure.figsize'] = (20, 12)
                plt.subplot(3, 1, 1)
                plt.plot(x, song_action_list, 'o', markersize=2)
                plt.xlabel('time / s')
                plt.yticks(range(0, args.num_clusters))
                plt.ylabel('action')
                song_num = fp.split('specs')[-1].split('.')[0][1:-4]
                plt.title('Song No. {}, f1: {}, score: {}'.format(song_num, info['f1'], song_score))

                plt.subplot(3, 1, 2)
                plt.plot(x, song_reward_list, 'o', markersize=2)
                plt.xlabel('time / s')
                plt.ylabel('reward')

                plt.subplot(3, 1, 3)
                plt.plot(x, label_list, 'o', markersize=2)
                plt.xlabel('time / s')
                plt.ylabel('label')
                #print(res_dir)
                plt.savefig(os.path.join(res_dir, song_num + '.jpg'))
                plt.close()

        score /= len(val_dataset)
        f1 /= len(val_dataset)

if __name__ == '__main__':
    args = get_args()
    test(args)
    # test_links(args)