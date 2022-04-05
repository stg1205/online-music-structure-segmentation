import argparse
from email.policy import default
import torch
from rl import rl_model, omss_env, rl_utils
from supervised_model import sup_model
from utils import config as cfg
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import os
import wandb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

def omss_train_val_test_split(val_pct, test_pct=None, test_idxs=None):
    files = os.listdir(cfg.SALAMI_DIR)
    fps = np.array(list(map(lambda x: os.path.join(cfg.SALAMI_DIR, x), files)))
    if test_idxs:
        test_dataset = fps[test_idxs]
        remain_idxs = np.setdiff1d(np.arange(len(files)), test_idxs)
        train_val_dataset = fps[remain_idxs]
    else:
        train_val_dataset, test_dataset = train_test_split(fps, test_size=test_pct, random_state=args.seed)
    
    train_dataset, val_dataset = train_test_split(train_val_dataset, test_size=val_pct, random_state=args.seed)
    
    return train_dataset, val_dataset, test_dataset


def dqn_loss(q_net, target_q_net, batch, gamma):
    state = state_to(batch["obs_batch"], device)
    next_state = state_to(batch["next_obs_batch"], device)
    action = batch["action_batch"].reshape(-1, 1).to(device)
    reward = batch["reward_batch"].reshape(-1, 1).to(device)
    done = batch["done_batch"].reshape(-1, 1).to(device)

    # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
    cur_q_value = q_net(state)['q'].gather(1, action)
    next_q_value = target_q_net(next_state)['q'].gather(  # Double DQN
            1, q_net(next_state)['q'].argmax(dim=1, keepdim=True)
        ).detach()
    mask = 1 - done
    target = (reward + gamma * next_q_value * mask).to(device)

    # calculate dqn loss
    loss = F.smooth_l1_loss(cur_q_value, target, reduction='none')
    
    return loss


def state_to(state, device):
    # add a batch dimension and move to device
    format_state = {}
    if isinstance(state['embedding_space'], torch.Tensor) :
        format_state['embedding_space'] = state['embedding_space'].unsqueeze(0).to(device)
        format_state['cur_chunk'] = state['cur_chunk'].unsqueeze(0).to(device)
        format_state['onehot_mask'] = state['onehot_mask'].unsqueeze(0).to(device)
    else:
        format_state['embedding_space'] = state['embedding_space'].to(device)
        format_state['cur_chunk'] = state['cur_chunk'].to(device)
        format_state['onehot_mask'] = state['onehot_mask'].to(device)
    return format_state

def state_out(out):
    out['q'] = out['q'].detach().cpu()
    out['new_feature'] = out['new_feature'].detach().cpu()


def train(args):
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initialize model and environment
    frontend_model = [sup_model.UnsupEmbedding((cfg.BIN, cfg.CHUNK_LEN)) for _ in range(2)]
    backend_input_size = cfg.EMBEDDING_DIM + args.num_clusters if args.cluster_encode else cfg.EMBEDDING_DIM
    backend_model = [rl_model.Backend(backend_input_size, 
                                        args.hidden_size, 
                                        args.num_layers,
                                        args.num_clusters
                                        #,args.num_heads)
                                        ) for _ in range(2)]  # TODO
    q_net = rl_model.QNet(frontend=frontend_model[0], backend=backend_model[0]).to(device)
    # load pretrained model
    if len(args.pretrained) > 3:
        q_net.load_frontend(args.pretrained)
        print('load pretrained frontend model!')
    # set target network to eval mode
    target_q_net = rl_model.QNet(frontend=frontend_model[1], backend=backend_model[1]).to(device)
    target_q_net.load_state_dict(q_net.state_dict())
    target_q_net.eval()
    # train validation test dataset (file paths)
    test_idxs = None
    if args.test_idxs:
        # load test set indexs TODO
        test_idxs = []
    train_dataset, val_dataset, test_dataset = omss_train_val_test_split(cfg.val_pct, cfg.test_pct, test_idxs)

    # optimizer
    optim = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    # use two buffer to calculate one step and n step losses
    buffer = rl_utils.ReplayBuffer(args.buffer_size, 1, args.priority, args.alpha, args.gamma)
    if args.n_step > 1:
        n_buffer = rl_utils.ReplayBuffer(args.buffer_size, args.n_step, args.priority, args.alpha, args.gamma)
    
    # log
    import time
    run_id = time.strftime("%m%d%H%M", time.localtime())
    # wandb.login(key='1dd98ff229fabf915050f551d8d8adadc9276b51')
    # wandb.init(project='online_mss', name=run_id, config=args)
    # wandb.config.update(args)
    exp_dir = cfg.RL_DIR + run_id
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # train loop
    print('training start...................')
    best_score = 0
    for i in range(args.epoch_num):
        eps = args.train_eps  # TODO: where to put these?
        beta = args.beta
        update_count = 0
        step_count = 0

        q_net.train()
        # iterate over train set
        for j, fp in enumerate(train_dataset):
            # start a new environment TODO: to save memory, load spectrogram every epoch or load all from training start?
            env = omss_env.OMSSEnv(frontend_model[0], 
                                    args.num_clusters, 
                                    fp, 
                                    cluster_encode=args.cluster_encode, 
                                    mode='train')  # TODO: use which frontend?
            state = env.make(device)
            done = False
            song_score = 0
            song_update_count = 0
            mean_loss = 0
            # step loop
            print('training {}th song............'.format(j))
            while not done:
                # take action and step
                format_state = state_to(state, device)  # has to contain a batch dimension
                out = q_net(format_state, padded=False)
                state_out(out)
                feature = out['new_feature']
                #print(feature.shape)
                # epsilon greedy policy
                # only can choose up to cur cluster number + 1
                if eps > np.random.random():
                    cur_mask = state['onehot_mask']
                    cur_max_cluster = torch.max(torch.argmax(cur_mask, dim=-1))
                    if cur_max_cluster + 1 < args.num_clusters:
                        upbound = cur_max_cluster + 2
                    else:
                        upbound = args.num_clusters
                    selected_action = torch.randint(0, upbound, (1,))
                else:
                    selected_action = torch.argmax(out['q'])
                action = {
                    'action': selected_action,
                    'new_feature': feature
                }
                next_state, reward, done, info = env.step(action)
                song_score += reward.item()

                # buffer
                if args.n_step > 1:
                    # this will return none if not reaching n_step or the first transition
                    one_step_transition = n_buffer.store(state, selected_action, reward, next_state, done)
                    if one_step_transition:
                        buffer.store(*one_step_transition)
                else:
                    buffer.store(state, selected_action, reward, next_state, done)
                state = next_state

                # buffer reach the required size, training is ready
                if len(buffer) >= args.batch_size:
                    print('update!')
                    batch = buffer.sample(args.batch_size, beta)
                    idxs = batch['idxs']
                    
                    # calculate loss by combining one step and n step loss
                    ele_wise_loss = dqn_loss(q_net, target_q_net, batch, args.gamma)
                    if args.n_step > 1:
                        n_batch = n_buffer.sample_from_idxs(idxs, beta)
                        gamma = args.gamma ** args.n_step
                        ele_wise_n_loss = dqn_loss(q_net, target_q_net, n_batch, gamma)
                        ele_wise_loss += ele_wise_n_loss
                    # print(ele_wise_loss)
                    if args.priority:
                        weights = batch['weights'].to(device)
                        loss = torch.mean(ele_wise_loss * weights)
                    else:
                        loss = torch.mean(ele_wise_loss)
                    # back propagation
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    update_count += 1   # TODO where to put this
                    mean_loss += loss.item()
                    # hard update target network
                    if update_count % args.target_update_freq == 0:
                        target_q_net.load_state_dict(q_net.state_dict())

                    # PER: update priorities  TODO not understand
                    loss_for_prior = ele_wise_loss.detach().cpu().numpy()
                    new_priorities = loss_for_prior + args.prior_eps
                    buffer.update_priorities(idxs, new_priorities)

                    song_update_count += 1

                # linearly decrease epsilon
                if eps > args.final_train_eps:
                    eps -= (args.train_eps - args.final_train_eps) * args.train_eps_decay
                
                # update beta
                if args.priority:
                    if step_count <= args.beta_anneal_step:
                        beta = args.beta - step_count / args.beta_anneal_step * \
                            (args.beta - args.final_beta)
                    else:
                        beta = args.final_beta
                    
            # reset buffer after iterate over one song
            buffer.reset()
            n_buffer.reset()
            # train_metrics = {
            #             'train/loss': mean_loss / song_update_count,
            #             'train/score': song_score,
            #             'train/epoch': (j + 1 + (len(train_dataset) * i)) / len(train_dataset)
            #         }
            # wandb.log(train_metrics)
        
        # val after one epoch
        q_net.eval()
        score = 0
        for k, fp in enumerate(val_dataset):
            env = omss_env.OMSSEnv(frontend_model[0], 
                                    args.num_clusters, 
                                    fp, 
                                    cluster_encode=args.cluster_encode, 
                                    mode='test')
            state = env.make().to(device)
            while not done:
                out = q_net(state)
                next_state, reward, done, info = env.step(out)
                state = next_state
                score += reward

        score /= len(val_dataset)
        # val_metrics = {
        #     'val/score': score
        # }
        # wandb.log(val_metrics)
        checkpoint = {
            'best_score': best_score,
            'state_dict': q_net.state_dict()
        }
        if score > best_score:
            checkpoint['best_score'] = score
            torch.save(checkpoint, os.path.join(exp_dir, "best_q_net.pth"))
        torch.save(checkpoint, os.path.join(exp_dir, "last_q_net.pth"))
    wandb.finish()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    so_far_best_pretrained = cfg.SUP_DIR + os.path.join('03020414', 'unsup_embedding_best.pt')

    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--test_idxs', type=str, default=None)
    # embedding model
    parser.add_argument('--pretrained', type=str, default=so_far_best_pretrained)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=1)
    # rl
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--cluster_encode', default=True)

    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--target_update_freq', type=int, default=100)
    ## eps greedy search
    parser.add_argument('--train_eps', type=float, default=1.)
    parser.add_argument('--final_train_eps', type=float, default=0.005)
    parser.add_argument('--train_eps_decay', type=float, default=1/2000)
    ## priority buffer
    parser.add_argument('--priority', type=bool, default=True)
    parser.add_argument('--prior_eps', type=float, default=1e-6)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--beta_anneal_step', type=int, default=2000)
    parser.add_argument('--final_beta', type=float, default=1.)
    ## buffer
    parser.add_argument('--buffer_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    ## n step
    parser.add_argument('--n_step', type=int, default=3)
    args = parser.parse_args()
    train(args)
