import argparse
import datetime
import os
from supervised_model.sup_model import Frontend
from utils import config as cfg
import time

import numpy as np
import torch
import wandb

from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv, DummyVectorEnv, SubprocVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.utils import TensorboardLogger, WandbLogger

from rl import tianshou_rl_model, tianshou_env
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def state_to(state, device, args):
    embedds = torch.as_tensor(state['embedding_space'][np.newaxis, ...]).to(device)
    centroids = torch.as_tensor(state['centroids'][np.newaxis, ...]).to(device)
    lens = state['lens'][np.newaxis, ...]
    if args.freeze_frontend:
        cur_embedding = torch.as_tensor(state['cur_embedding'][np.newaxis, ...]).to(device)
        return {
            'embedding_space': embedds,
            'cur_embedding': cur_embedding,
            'centroids': centroids,
            'lens': lens
        }
    else:
        cur_chunk = torch.as_tensor(state['cur_chunk'][np.newaxis, ...]).to(device)
        return {
            'embedding_space': embedds,
            'cur_chunk': cur_chunk,
            'centroids': centroids,
            'lens': lens
        }

def get_args():
    parser = argparse.ArgumentParser()

    # experiment set up
    parser.add_argument('--name', type=str)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default=None
    )

    # ----------------RL------------------
    # embedding model
    parser.add_argument('--freeze_frontend', action='store_true')
    
    # env
    parser.add_argument('--final_punish', type=float, default=-2.)
    parser.add_argument('--knowing_cluster_num', action='store_true')
    # backend
    parser.add_argument('--cluster_encode', action='store_true')
    parser.add_argument('--hidden_size', type=int, default=128)  #*
    parser.add_argument('--num_layers', type=int, default=1)    #*
    parser.add_argument('--num_heads', type=int, default=1)   # *
    parser.add_argument('--seq_max_len', type=int, default=128)
    parser.add_argument('--num_clusters', type=int, default=5)  # *
    parser.add_argument('--use_rnn', action='store_true')

    # training
    parser.add_argument("--epoch_num", type=int, default=100)
    parser.add_argument('--train_env_batch_size', type=int, default=4)
    parser.add_argument("--scale-obs", type=int, default=0)  # TODO
    parser.add_argument("--eps-test", type=float, default=0.)
    parser.add_argument("--eps-train", type=float, default=1.)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument('--eps_decay', type=float, default=1/1e6)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0000625)
    parser.add_argument("--gamma", type=float, default=0.99)
    # priority buffer
    parser.add_argument("--no-priority", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-final", type=float, default=1.)
    parser.add_argument("--beta-anneal-step", type=int, default=1000000)
    parser.add_argument("--no-weight-norm", action="store_true", default=False)
    parser.add_argument("--n-step", type=int, default=3)
    # dqn
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    
    return parser.parse_args()

def rm_invalid_mel_fp(files):
    valid_files = []
    for f in files:
        if f.startswith('.') or not f.endswith('npy'):
            continue
        valid_files.append(f)
    
    return valid_files

def omss_train_val_test_split(val_pct, test_pct, test_idxs, args):
    mel_dir = os.path.join(cfg.SALAMI_DIR, 'internet_melspecs')
    files = os.listdir(mel_dir)
    files = rm_invalid_mel_fp(files)
    fps = np.array(list(map(lambda x: os.path.join(mel_dir, x), files)))
    if test_idxs:
        test_dataset = fps[test_idxs]
        remain_idxs = np.setdiff1d(np.arange(len(files)), test_idxs)
        train_val_dataset = fps[remain_idxs]
    else:
        train_val_dataset, test_dataset = train_test_split(fps, test_size=test_pct, random_state=args.seed)
    
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=val_pct, random_state=args.seed)
    
    return train_dataset, val_dataset, test_dataset

def omss_train_val_split(val_pct, val_files, args):
    if cfg.dataset == 'salami':
        mel_dir = os.path.join(cfg.SALAMI_DIR, 'internet_melspecs')
    elif cfg.dataset == 'harmonix':
        mel_dir = os.path.join(cfg.HARMONIX_DIR, 'melspecs')

    files = os.listdir(mel_dir)
    files = rm_invalid_mel_fp(files)
    if val_files is not None:
        train_files = np.setdiff1d(files, val_files, assume_unique=True)
    else:
        train_files, val_files = train_test_split(files, test_size=val_pct, random_state=args.seed)
    train_dataset = np.array(list(map(lambda x: os.path.join(mel_dir, x), train_files)))
    val_dataset = np.array(list(map(lambda x: os.path.join(mel_dir, x), val_files)))

    return train_dataset, val_dataset

def validation(policy: DQNPolicy, val_dataset, args, frontend=None):
    q_net = policy.model
    q_net.eval()
    if not args.freeze_frontend:
        frontend = q_net.get_frontend()
    
    score = 0
    f1 = 0
    count = len(val_dataset)
    with torch.no_grad():
        with trange(len(val_dataset)) as t:
            for k in t:
        #for k in tqdm(range(len(val_dataset))):
                # if k < 25:
                #     continue
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
                # if not env.check_anno():
                #     count -= 1
                #     continue
                state = env.reset()
                done = False
                while not done:
                    format_state = state_to(state, device, args=args)
                    logits = policy.model(format_state)[0].detach().cpu().numpy()
                    # print(logits)
                    action = np.argmax(logits)
                    # print(action)
                    
                    # action = policy.take_action(state, env, args.test_eps, args.num_clusters)
                    next_state, reward, done, info = env.step(action)
                    # if args.logger:
                    #     wandb.log({
                    #         'val/action': action,
                    #         'val/reward': reward})
                    state = next_state
                    score += reward
                f1 += info['f1']
                t.set_description('f1: {}'.format(info['f1']))
                # print(reward.item())
        score /= count
        f1 /= count
    
    return score, f1


def train(args=get_args()):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # prepare dataset (file paths)
    test_csv = cfg.test_csv
    if test_csv:
        # load test set indexs
        import pandas as pd
        test_files = np.array(pd.read_csv(test_csv, header=None)[0])
    else:
        test_files = None
    # train_dataset, val_dataset, test_dataset = omss_train_val_test_split(cfg.val_pct, cfg.test_pct, test_idxs, args)
    train_dataset, val_dataset = omss_train_val_split(cfg.val_pct, test_files, args)
    print(len(train_dataset))

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

    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            
    # define policy
    policy = DQNPolicy(
        model=net,
        optim=optim,
        discount_factor=args.gamma,
        target_update_freq=args.target_update_freq,
        is_double=True
    ).to(device)

    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    if args.no_priority:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=args.train_env_batch_size,
            ignore_obs_next=True,
        )
    else:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=args.train_env_batch_size,
            ignore_obs_next=True,
            alpha=args.alpha,
            beta=args.beta,
            weight_norm=not args.no_weight_norm
        )

    # log
    run_id = time.strftime("%m%d%H%M", time.localtime())
    exp_dir = os.path.join(cfg.RL_EXP_DIR, run_id)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # logger
    if args.logger:
        if args.logger == "wandb":
            wandb.login(key='1dd98ff229fabf915050f551d8d8adadc9276b51')
            logger = WandbLogger(
                save_interval=1,
                name=args.name,
                run_id=run_id,
                config=args,
                project='online_mss',
                update_interval=100
            )
        writer = SummaryWriter(exp_dir)
        writer.add_text("args", str(args))
        if args.logger == "tensorboard":
            logger = TensorboardLogger(writer)
        else:  # wandb
            logger.load(writer)

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1 / args.eps_decay:
            eps = args.eps_train - env_step * args.eps_decay * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final

        policy.set_eps(eps)
        train_envs.set_env_attr('_eps', eps)

        if args.logger:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        if not args.no_priority:
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * \
                    (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buffer.set_beta(beta)
            if args.logger:
                logger.write("train/env_step", env_step, {"train/beta": beta})
    
    # load a previous policy
    if args.resume_path:
        checkpoint = torch.load(args.resume_path, map_location=device)
        try:
            policy.load_state_dict(checkpoint['state_dict'])
        except:
            # just load backend parameters
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('model_old'):
                    state_dict[k.replace('model_old', 'model_old._backend')] = state_dict.pop(k)
                else:
                    state_dict[k.replace('model', 'model._backend')] = state_dict.pop(k)  
            policy.load_state_dict(state_dict, strict=False)
        
        best_score = checkpoint['best_score']
        best_f1 = checkpoint['best_f1']
        print('best score: ', best_score)
        print("Loaded agent from: ", args.resume_path)
    else:
        best_score = 0
        best_f1 = 0
    
    gradient_step = 0
    env_step = 0
    # train loop
    for epoch in range(args.epoch_num):
        # iterate over train set
        # np.random.shuffle(train_dataset)
        env_batch = []
        batch_count = 1
        train_score = 0
        train_loss = 0
        
        with trange(len(train_dataset)) as t:
        # for j in tqdm(range(len(train_dataset))):
            for j in t:
                # continue
                # if j < 20:
                #     continue
                # prepare batch envs
                fp = train_dataset[j]
                env_batch.append(fp)
                print(fp)

                if not args.freeze_frontend:
                    frontend = net.get_frontend()

                # TODO ugly, but would be removed after washing dataset
                # env = tianshou_env.OMSSEnv(frontend,   # TODO cpu device?
                #                         args.num_clusters, 
                #                         fp, 
                #                         args.seq_max_len,
                #                         cluster_encode=args.cluster_encode, 
                #                         mode='train')
                # if env.check_anno():
                #     env_batch.append(fp)
                ######################################################

                if j != len(train_dataset)-1 and len(env_batch) < args.train_env_batch_size:
                    continue

                train_envs = DummyVectorEnv([lambda x=fp: tianshou_env.OMSSEnv(frontend,
                                        args.num_clusters, 
                                        x, 
                                        args.seq_max_len,
                                        knowing_cluster_num=args.knowing_cluster_num,
                                        final_eps=args.eps_train_final, 
                                        final_punish=args.final_punish, 
                                        cluster_encode=args.cluster_encode, 
                                        freeze_frontend=args.freeze_frontend,
                                        mode='train') for fp in env_batch])
                env_batch = []
                train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
                # print(train_envs)
                score = 0
                loss = 0
                count = 0

                # collect one episode
                train_fn(epoch, env_step)
                policy.eval()
                coll_res = train_collector.collect(n_episode=args.train_env_batch_size)
                policy.train()
                if not args.freeze_frontend:
                    policy.model._freeze_bm()
                t.set_description('Epoch:[{}/{}], reward:{:.5f}, n_st:{}'.format(epoch, args.epoch_num, coll_res['rew'], coll_res['n/st']))
                
                # log train data
                env_step += coll_res['n/st']
                if args.logger:
                    logger.log_train_data(coll_res, env_step)
                
                train_score += coll_res['rew']  # mean reward

                # increase batch size with buffer size
                perc = 1 + len(buffer) / args.buffer_size
                batch_size = round(perc * args.batch_size)
                update_times = round(perc * args.update_per_step * coll_res['n/st'])
                for _ in range(update_times):
                    losses = policy.update(batch_size * args.train_env_batch_size, buffer)
                    gradient_step += 1
                    if args.logger:
                        logger.log_update_data(losses, gradient_step)
                    train_loss += losses['loss']
                
                # update frontend if needed
                if not args.freeze_frontend:
                    train_envs.set_env_attr('_frontend_model', net.get_frontend())

                batch_count += 1

                ## step wise collection
                # while True:
                #     # eps, beta linearly decay
                #     train_fn(epoch, env_step)
                #     # collect step data
                #     coll_res = train_collector.collect(n_step=args.batch_size * args.train_env_batch_size)
                #     if coll_res['n/ep'] > 0:
                #         score += coll_res['rew'] * coll_res['n/ep'] # TODO not include the rewards of some unfinished episodes 
                    
                #     print(coll_res['n/st'])
                #     # update policy
                #     for _ in range(round(10)):
                #         update_res = policy.update(args.batch_size * args.train_env_batch_size, buffer)  # TODO do more training
                #         loss += update_res['loss']
                #         count += 1

                #     # update frontend if needed
                #     if not args.freeze_frontend:
                #         train_envs.set_env_attr('_frontend_model', net.get_frontend())
                    
                #     if train_collector.collect_episode >= args.train_env_batch_size:  # TODO should be when the longest on ends, or just count the episodes
                #     # if train_collector.collect_step >= args.train_collect_steps: # TODO??
                #         train_score += score / train_collector.collect_episode  # score per episode
                #         train_loss += loss / count
                #         batch_count += 1
                #         break
                    
                #     print(train_collector.collect_step)
                #     env_step += train_collector.collect_step
            
                
        train_score /= batch_count
        train_loss /= batch_count
        # validation
        if not args.freeze_frontend:
            val_score, f1 = validation(policy, val_dataset, args)
        else:
            val_score, f1 = validation(policy, val_dataset, args, frontend)
        # log validation metrics
        if args.logger:
            metrics = {'val/val_score': val_score, 
                                'val/f1': f1,
                                'val/train_loss': train_loss,
                                'val/train_score': train_score}
            wandb.log(metrics)
        # save model
        checkpoint = {
            'best_score': best_score,
            'best_f1': best_f1,
            'state_dict': policy.state_dict()
        }
        #print(score)
        if val_score > best_score:
            checkpoint['best_score'] = val_score
            best_score = val_score
            torch.save(checkpoint, os.path.join(exp_dir, "best_score_policy.pth"))
        if f1 > best_f1:
            checkpoint['best_f1'] = f1
            best_f1 = f1
            torch.save(checkpoint, os.path.join(exp_dir, 'best_f1_policy.pth'))
        torch.save(checkpoint, os.path.join(exp_dir, "last_policy.pth"))

if __name__ == "__main__":
    train(get_args())
