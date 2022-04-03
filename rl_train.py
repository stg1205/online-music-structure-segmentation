import argparse
import torch
from rl import rl_model, omss_env
from supervised_model import sup_model
from utils import config as cfg
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.policy import RainbowPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.env import ShmemVectorEnv, SubprocVectorEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_train_environments(args):
    train
    cfg.SALAMI_DIR



def stop_fn(mean_rewards):
    if env.spec.reward_threshold:
        return mean_rewards >= env.spec.reward_threshold
    elif "Pong" in args.task:
        return mean_rewards >= 20
    else:
        return False

def train(args):
    # initialize model and environment
    frontend_model = sup_model.UnsupEmbedding((cfg.BIN, cfg.CHUNK_LEN))
    backend_model = rl_model.Backend(cfg.EMBEDDING_DIM, 
                                    args.num_atoms,
                                    args.noisy_std,
                                    is_dueling=args.dueling,
                                    is_noisy=args.noisy)  # TODO
    q_net = rl_model.QNet(frontend=frontend_model, backend=backend_model)
    if args.pretrained:
        q_net.load_frontend(args.pretrained)
    env = omss_env.OMSSEnv(frontend_model, args.num_clusters)

    # seed
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    # log
    if args.logger == "wandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_dir)
    writer.add_text('args', str(args))
    logger.load(writer)


    # policy and optimizer
    optim = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    policy = RainbowPolicy(
        q_net,
        optim,
        args.gamma,
        args.num_atoms,
        args.v_min,
        args.v_max,
        args.n_step,
        target_update_freq=args.target_update_freq).to(device)
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))
    
    def stop_fn(mean_rewards):
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        elif "Pong" in args.task:
            return mean_rewards >= 20
        else:
            return False

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = PrioritizedVectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack,
                alpha=args.alpha,
                beta=args.beta
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=args.test_num, render=args.render
            )
        rew = result["rews"].mean()
        print(f"Mean reward (over {result['n/ep']} episodes): {rew}")

    if args.watch:
        watch()
        exit(0)

    train_dataset, test_dataset = train_test_split(args.train_env_batch_num, args.test_env_batch_num)
    test_envs = make_environments(args, test_dataset)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    
    if args.priority:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            ignore_obs_next=True,
            save_only_last_obs=True,
            stack_num=args.frames_stack,
            alpha=args.alpha,
            beta=args.beta,
            weight_norm=not args.no_weight_norm
        )
            
    else:
        buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(train_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack
            )

    # train loop
    
    for i in range(args.epoch_num):
        # iterate over train set
        
        for train_env_batch in train_dataset:
            train_envs = make_environments(args, train_env_batch)
            # collector
            train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
            collect_result = train_collector.collect(n_episode=args.n_episode)
             # nature DQN setting, linear decay in the first 1M steps
            if env_step <= 1e6:
                eps = args.eps_train - env_step / 1e6 * \
                    (args.eps_train - args.eps_train_final)
            else:
                eps = args.eps_train_final
            policy.set_eps(eps)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/eps": eps})
            if args.priority:
                if env_step <= args.beta_anneal_step:
                    beta = args.beta - env_step / args.beta_anneal_step * \
                        (args.beta - args.beta_final)
                else:
                    beta = args.beta_final
                buffer.set_beta(beta)
                if env_step % 1000 == 0:
                    logger.write("train/env_step", env_step, {"train/beta": beta})

            losses = policy.update(args.sample_size, train_collector.buffer)

            buffer.reset()
    




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--train_env_batch_num', type=int)
    parser.add_argument('--test_env_batch_num', type=int)
    parser.add_argument('--n_episode', type=int)
    parser.add_argument('--sample_size', type=int)
    parser.add_argument('--train_eps', type=float)
    parser.add_argument('--test_eps', type=float)
    parser.add_argument('--priority', type=bool)
    parser.add_argument('--buffer_size', type=int)
    args = parser.parse_args()
    train(args)
