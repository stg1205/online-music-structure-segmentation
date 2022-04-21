import argparse
import torch
from rl import rl_model, omss_env, rl_utils
from utils import config as cfg
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import os
import wandb
from tqdm import tqdm
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def omss_train_val_test_split(val_pct, test_pct=None, test_idxs=None):
    mel_dir = os.path.join(cfg.SALAMI_DIR, 'internet_melspecs')
    files = os.listdir(mel_dir)
    fps = np.array(list(map(lambda x: os.path.join(mel_dir, x), files)))
    if test_idxs:
        test_dataset = fps[test_idxs]
        remain_idxs = np.setdiff1d(np.arange(len(files)), test_idxs)
        train_val_dataset = fps[remain_idxs]
    else:
        train_val_dataset, test_dataset = train_test_split(fps, test_size=test_pct, random_state=args.seed)
    
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=val_pct, random_state=args.seed)
    
    return train_dataset, val_dataset, test_dataset


def validation(q_net, policy, val_dataset, args):
    q_net.eval()
    score = 0
    f1 = 0
    count = len(val_dataset)
    with torch.no_grad():
        for k in tqdm(range(len(val_dataset))):
            fp = val_dataset[k]
            env = omss_env.OMSSEnv(#q_net.module.get_frontend(), 
                                    q_net.get_frontend(),
                                    args.num_clusters, 
                                    fp, 
                                    args.seq_max_len,  # TODO don't need this in val
                                    cluster_encode=args.cluster_encode, 
                                    mode='test')
            if not env.check_anno():
                count -= 1
                continue
            state = env.make()
            done = False
            while not done:
                action = policy.take_action(state, env, args.test_eps, args.num_clusters)
                print(action['action'])
                next_state, reward, done, info = env.step(action)
                state = next_state
                score += reward.item()
            f1 += reward.item()
            # print(reward.item())
        score /= count
        f1 /= count
    q_net.train()
    return score, f1


def train(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize model
    gpus = [0, 1, 2, 3]
    backend_input_size = cfg.EMBEDDING_DIM + args.num_clusters if args.cluster_encode else cfg.EMBEDDING_DIM
    nets = [rl_model.QNet(input_shape=(cfg.BIN, cfg.CHUNK_LEN),
                            embedding_size=backend_input_size,
                            hidden_size=args.hidden_size,
                            num_layers=args.num_layers,
                            num_heads=args.num_heads,
                            num_clusters=args.num_clusters,
                            cluster_encode=args.cluster_encode,
                            use_rnn=args.use_rnn,
                            freeze_frontend=args.freeze_frontend).to(device) for _ in range(2)]
    q_net = nets[0]
    # load pretrained model
    if len(args.pretrained) > 3:
        q_net.load_frontend(args.pretrained)
        print('load pretrained frontend model!')
    elif args.resume_path:
        checkpoint = torch.load(args.resume_path)
        print(checkpoint['best_score'])
        q_net.load_state_dict(checkpoint['state_dict'])
        print('load best q net!')
    # set target network to eval mode
    target_q_net = nets[1]
    target_q_net.load_state_dict(q_net.state_dict())
    target_q_net.eval()
    if args.parallel:
        # q_net, target_q_net = [nn.parallel.DistributedDataParallel(
        q_net, target_q_net = [nn.DataParallel(
            net, device_ids=gpus, output_device=gpus[0]) for net in (q_net, target_q_net)]

    # prepare dataset (file paths)
    test_idxs = None
    if args.test_idxs:
        # load test set indexs TODO
        test_idxs = []
    train_dataset, val_dataset, test_dataset = omss_train_val_test_split(cfg.val_pct, cfg.test_pct, test_idxs)

    # optimizer
    if args.freeze_frontend:
        optim = torch.optim.Adam(q_net._backend.parameters(), lr=args.lr)
    else:
        optim = torch.optim.Adam(q_net.parameters(), lr=args.lr)

    # policy
    policy = rl_utils.Policy(q_net=q_net, 
                            target_q_net=target_q_net, 
                            gamma=args.gamma,
                            target_update_freq=args.target_update_freq,
                            n_step=args.n_step,
                            optim=optim,
                            device=device)

    # use two buffer to calculate one step and n step losses
    buffer = rl_utils.ReplayBuffer(args.buffer_size, 
                                    args.num_clusters,
                                    backend_input_size,
                                    1, args.priority, args.alpha, args.gamma)
    if args.n_step > 1:
        n_buffer = rl_utils.ReplayBuffer(args.buffer_size, 
                                        args.num_clusters,
                                        backend_input_size,
                                        args.n_step, args.priority, args.alpha, args.gamma)
    
    # log
    run_id = time.strftime("%m%d%H%M", time.localtime())
    if args.wandb:
        wandb.login(key='1dd98ff229fabf915050f551d8d8adadc9276b51')
        wandb.init(project='online_mss', name=run_id, config=args)
        wandb.config.update(args)
    exp_dir = os.path.join(cfg.RL_EXP_DIR, run_id)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # train loop
    best_score = 0
    for i in range(args.epoch_num):
        eps = args.train_eps  # TODO: where to put these?
        beta = args.beta
        step_count = 0
        train_score = 0
        # iterate over train set
        np.random.shuffle(train_dataset)
        for j in tqdm(range(len(train_dataset))):
            continue
            fp = train_dataset[j]
            print(fp)
            # start a new environment TODO: to save memory, load spectrogram every epoch or load all from training start?
            env = omss_env.OMSSEnv(#q_net.module.get_frontend(), 
                                    q_net.get_frontend(), 
                                    args.num_clusters, 
                                    fp, 
                                    args.seq_max_len,
                                    cluster_encode=args.cluster_encode, 
                                    mode='train')  # TODO: use which frontend?
            if not env.check_anno():
                continue
            state = env.make()
            done = False
            song_score = 0
            song_update_count = 0
            mean_loss = 0
            # step loop
            tic = time.time()
            while not done:                         
                # take action
                action = policy.take_action(state, env, eps, args.num_clusters)
                selected_action = action['action']
                # take a step
                next_state, reward, done, info = env.step(action)
                song_score += reward.item()
                # print(song_score)

                # add transition to buffer
                if args.n_step > 1:
                    # this will return none if not reaching n_step else the first transition
                    one_step_transition = n_buffer.store(state, selected_action, reward, next_state, done)
                    if one_step_transition:
                        buffer.store(*one_step_transition)
                else:
                    buffer.store(state, selected_action, reward, next_state, done)
                state = next_state

                # when buffer reaches the required size, training is ready
                if len(buffer) >= args.batch_size:
                    batch = buffer.sample(args.batch_size, beta)
                    idxs = batch['idxs']
                    n_batch = n_buffer.sample_from_idxs(idxs, beta) if args.n_step > 1  else None

                    ele_wise_loss, loss = policy.update(batch, n_batch, optim)
                    mean_loss += loss.item()
                    
                    if args.priority:
                        # PER: update priorities  TODO not understand
                        loss_for_prior = ele_wise_loss.detach().cpu().numpy()
                        #print(loss_for_prior)
                        new_priorities = loss_for_prior + args.prior_eps
                        buffer.update_priorities(idxs, new_priorities)

                    song_update_count += 1

                # linearly decrease epsilon
                if eps > args.final_train_eps:
                    eps -= (args.train_eps - args.final_train_eps) * args.train_eps_decay
                
                print(step_count, eps)
                # update beta
                if args.priority:
                    if step_count <= args.beta_anneal_step:
                        beta = args.beta - step_count / args.beta_anneal_step * \
                            (args.beta - args.final_beta)
                    else:
                        beta = args.final_beta
                if args.wandb:
                    wandb.log({
                            'explore/action': selected_action,
                            'explore/reward': reward,
                            'explore/eps': eps,
                            'explore/beta': beta})
                step_count += 1

            toc = time.time()
            print('an episode takes {}s'.format(toc-tic))
            # reset buffer after iterate over one song  TODO: maybe not useful
            # buffer.reset()
            # n_buffer.reset()
            train_score += song_score
            if args.wandb:
                # record the metric for every song in an epoch
                episode_metrics = {
                            'episode/loss': mean_loss / song_update_count,
                            'episode/score': song_score,
                            'episode/epoch': (j + 1 + (len(train_dataset) * i)) / len(train_dataset)
                        }
                wandb.log(episode_metrics)
        
        # val after one epoch
        score, f1 = validation(q_net, policy, val_dataset, args)
        # print(score, f1)
        if args.wandb:
            val_metrics = {
                'val/train_score': train_score,
                'val/score': score,
                'val/f1': f1
            }
            wandb.log(val_metrics)
        checkpoint = {
            'best_score': best_score,
            'state_dict': q_net.state_dict()
        }
        #print(score)
        if f1 > best_score:
            checkpoint['best_score'] = f1
            best_score = f1
            torch.save(checkpoint, os.path.join(exp_dir, "best_q_net.pth"))
        torch.save(checkpoint, os.path.join(exp_dir, "last_q_net.pth"))
    wandb.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    so_far_best_pretrained = os.path.join(cfg.SUP_EXP_DIR, '03020414', 'unsup_embedding_best.pt')

    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--freeze_frontend', action='store_true')
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
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--seq_max_len', type=int, default=128)
    parser.add_argument('--num_clusters', type=int, default=5)  # *
    parser.add_argument('--cluster_encode', action='store_true')
    #parser.add_argument('--max_grad_norm', type=float, default=2.0)
    
    parser.add_argument('--use_rnn', action='store_true')

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
    train(args)
