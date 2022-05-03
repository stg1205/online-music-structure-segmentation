from ast import arg
import time
import os
from argparse import ArgumentParser
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, random_split
import torch
from pytorch_metric_learning.losses import MultiSimilarityLoss
from utils.msaf_validation import scluster_eval

from supervised_model.mss_data import HarmonixDataset, SongDataset
from supervised_model.sup_model import Frontend
import utils.config as cfg
from utils.logger import create_logger
from utils.modules import split_to_chunk_with_hop

from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(model, val_loader, criterion, args):
    '''
    validate the MSS performance using MSA algorithm
    '''
    score = 0
    num = 0
    loss_sum = 0
    score_sum = 0
    seg_metric = {'HitRate_0.5F': 0, 
                'HitRate_3F': 0,
                'PWF': 0,
                'Sf': 0}
    model.eval()
    with torch.no_grad():
        with trange(len(val_loader)) as t:
            iterator = iter(val_loader)
            for i in t:
                # if i > 2:
                #     continue
                # forward
                batch = next(iterator)
                song, ref_times, ref_labels = batch['data'].squeeze(0), \
                                                    batch['ref_times'].squeeze(0), \
                                                    batch['ref_labels'].squeeze(0)
                # tic = time.time()
                song_dataset = SongDataset(song, ref_times, ref_labels, args.val_batch_size, mode='val')
                song_loader = DataLoader(song_dataset, args.val_batch_size, drop_last=False)
                # print('loader {}s'.format(time.time()-tic))
                # tic = time.time()
                for j, (chunks, chunk_labels) in enumerate(song_loader):
                    chunks = chunks.to(device)
                    chunk_labels = chunk_labels.to(device)
                    
                    batch_embeddings = model(chunks)
                    loss = criterion(batch_embeddings, chunk_labels)
                    loss_sum += loss.item()
                    num += 1

                    if j == 0:
                        embeddings = batch_embeddings
                    else:
                        embeddings = torch.concat([embeddings, batch_embeddings], dim=0)
                # print(j)
                # print('infer {}s'.format(time.time() - tic))
                # evaluation
                song_embedding = torch.transpose(embeddings, 0, 1)
                song_embedding = song_embedding.cpu().detach().numpy()
                ref_times = ref_times.cpu().detach().numpy()
                ref_labels = ref_labels.cpu().detach().numpy()
                res = scluster_eval(song_embedding, ref_times, ref_labels[:-1])
                score = 5/14*res['HitRate_0.5F'] + 2/14*res['HitRate_3F'] + 4/14*res['PWF'] + 3/14*res['Sf']
                score_sum += score
                for k in seg_metric:
                    seg_metric[k] += res[k]

                t.set_description(
                    'Song:[{}/{}], loss={:.5f}, score={:.3f}'.format(i, len(val_loader), loss.item(), score))

    loss_avg = loss_sum / num
    score_avg = score_sum / len(val_loader)
    for k in seg_metric:
        seg_metric[k] /= len(val_loader)
    
    return score_avg, loss_avg, seg_metric


def experiment_setup(args):
    experiment_id = time.strftime("%m%d%H%M", time.localtime())
    exp_dir = os.path.join(cfg.SUP_EXP_DIR, experiment_id)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    # log
    log_fp = exp_dir + '/train.log'
    logger = create_logger(log_fp)
    logger.info('Start training!')
    logger.info(args)
    logger.info('chunk size: {}, train hop size: {}, eval hop size: {}'.format(cfg.CHUNK_LEN, cfg.train_hop_size, cfg.eval_hop_size))
    return exp_dir, logger, experiment_id


def train(args):
    '''
    train function
    '''
    # hyperparameters
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    batch_size = args.batch_size
    lr = args.lr
    n_epochs = args.n_epochs

    # models
    best_score = 0
    min_val_loss = np.Inf
    if args.model == 'unsup_embedding':
        model = Frontend((cfg.BIN, cfg.CHUNK_LEN), embedding_dim=args.embedding_dim)

    # loss, dataloader, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, mode='min', patience=3)
    criterion = MultiSimilarityLoss()
    dataset = HarmonixDataset()
    # train, test split
    train_len = int(len(dataset) * 0.75)
    test_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, lengths=(train_len, test_len), 
                                generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set)

    # set experiment environment
    exp_dir, logger, experiment_id = experiment_setup(args)
    writer = SummaryWriter(log_dir=os.path.join('runs', experiment_id))
    if args.pretrained:
        logger.info("Load pretrained model: " + args.pretrained)
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        best_score = checkpoint['best_score']
    print(model)
    model.to(device)

    # train loop
    for epoch in range(n_epochs):
        batch_sum, loss_sum = 0, 0
        model.train()
        iterator = iter(train_loader)
        with trange(len(train_loader)) as t:
            for _ in t:
                # continue
                batch = next(iterator)
                song, times, labels = batch['data'], batch['ref_times'], batch['ref_labels']
                song, labels = song.squeeze(0), labels.squeeze(0)

                song_dataset = SongDataset(song, times, labels, batch_size, mode='train')
                song_loader = DataLoader(song_dataset, batch_size=batch_size)

                for k, (examples, example_labels) in enumerate(song_loader):
                    # batch sample in a song
                    # forward
                    examples = examples.to(device)
                    # print(examples.shape)
                    example_labels = example_labels.to(device)
                    # print(example_labels.shape)
                    embeddings = model(examples)
                    loss = criterion(embeddings, example_labels)

                    loss_sum += loss.item()
                    batch_sum += 1
                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                t.set_description(
                    'Epoch:[{}/{}], loss={:.5f}'.format(epoch, n_epochs, loss.item()))
        # validate
        score_avg, val_loss_avg, seg_metric = validate(model, val_loader, criterion, args)
        checkpoint = {'state_dict': model.state_dict(),
                      'best_score': best_score}
        scheduler.step(score_avg)
        # save the best model
        if score_avg > best_score:
            best_score = score_avg
            checkpoint['best_score'] = best_score
            torch.save(checkpoint, os.path.join(exp_dir, args.model+'_best.pt'))
        # save the latest model
        torch.save(checkpoint, os.path.join(exp_dir, args.model+'_last.pt'))

        loss_avg = loss_sum/batch_sum if batch_sum != 0 else 0
        logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_loss={:.5f}\t score={:.3f}\t best_score{:.3f}'.format(
            epoch, n_epochs, loss_avg, val_loss_avg, score_avg, best_score))
        writer.add_scalar('train/train_loss', loss_avg, epoch)
        writer.add_scalar('val/val_loss', val_loss_avg, epoch)
        writer.add_scalar('val/val_score', score_avg, epoch)
        for k in seg_metric:
            writer.add_scalar('val/{}'.format(k), seg_metric[k], epoch)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--model', type=str,
                        default='unsup_embedding', choices=['unsup_embedding'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrained', type=str,
                        help='the path of pre-trained model')
    parser.add_argument('--embedding_dim', type=int)

    args = parser.parse_args()
    train(args)
