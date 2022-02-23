import time
import os
from argparse import ArgumentParser
from math import ceil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from tqdm import trange
from torch.utils.data import DataLoader, random_split
import torch
from pytorch_metric_learning.losses import MultiSimilarityLoss
from utils.msaf_validation import scluster

from supervised_model.mss_data import HarmonixDataset, SongDataset
from supervised_model.sup_model import UnsupEmbedding
import utils.config as cfg
from utils.logger import create_logger
from utils.modules import split_to_chunk_with_hop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(model, val_loader, criterion):
    '''
    validate the MSS performance using MSA algorithm
    '''
    score = 0
    num = 0
    loss_sum = 0
    score_sum = 0
    model.eval()
    with torch.no_grad():
        with trange(len(val_loader)) as t:
            iterator = iter(val_loader)
            for i in t:
                # forward
                batch = next(iterator)
                song, chunk_labels, ref_times, ref_labels = batch['data'].squeeze(0), \
                                                    batch['chunk_labels'].squeeze(0), \
                                                    batch['ref_times'].squeeze(0), \
                                                    batch['ref_labels'].squeeze(0)
                # adjust input dimension, group chunks to a batch
                song = split_to_chunk_with_hop(song)
                song, chunk_labels = song.to(device), chunk_labels.to(device)
                embeddings = model(song)
                loss = criterion(embeddings, chunk_labels)
                loss_sum += loss.item()
                num += 1

                # evaluation
                song_embedding = torch.transpose(embeddings, 0, 1)
                song_embedding = song_embedding.cpu().detach().numpy()
                ref_times = ref_times.cpu().detach().numpy()
                ref_labels = ref_labels.cpu().detach().numpy()
                res = scluster(song_embedding, ref_times, ref_labels)
                score = 5/14*res['HitRate_0.5F'] + 2/14*res['HitRate_3F'] + 4/14*res['PWF'] + 3/14*res['Sf']
                score_sum += score

                t.set_description(
                    'Song:[{}/{}], loss={:.5f}, score={:.3f}'.format(i, len(val_loader), loss.item(), score))

    loss_avg = loss_sum / num
    score_avg = score_sum / num
    return score_avg, loss_avg


def experiment_setup(args):
    experiment_id = time.strftime("%m%d%H%M", time.localtime())
    exp_dir = '/content/drive/Othercomputers/MyEnvy/online-music-structure-segmentation/supervised_model/sup_experiments/' + experiment_id
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    # log
    log_fp = exp_dir + '/train.log'
    logger = create_logger(log_fp)
    logger.info('Start training!')
    logger.info(args)
    return exp_dir, logger


def train(args):
    '''
    train function
    '''
    # hyperparameters
    torch.manual_seed(args.seed)
    batch_size = args.batch_size
    lr = args.lr
    n_epochs = args.n_epochs

    # models
    best_score = 0
    min_val_loss = np.Inf
    if args.model == 'unsup_embedding':
        model = UnsupEmbedding((cfg.BIN, cfg.CHUNK_LEN))

    # loss, dataloader, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MultiSimilarityLoss()
    dataset = HarmonixDataset()
    # train, test split
    train_len = int(len(dataset) * 0.75)
    test_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, lengths=(train_len, test_len))
    train_loader = DataLoader(train_set, shuffle=True)
    val_loader = DataLoader(val_set)

    # set experiment environment
    exp_dir, logger = experiment_setup(args)
    if args.pretrained:
        logger.info("Load pretrained model: " + args.pretrained)
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        best_score = checkpoint['best_score']
    model.to(device)
    for epoch in range(n_epochs):
        batch_sum, loss_sum = 0, 0
        model.train()
        iterator = iter(train_loader)
        with trange(len(train_loader)) as t:
            for _ in t:
                batch = next(iterator)
                song, labels = batch['data'], batch['chunk_labels']
                song, labels = song.squeeze(0), labels.squeeze(0)
                # complement the last batch TODO: shuffle first
                num_batch = len(labels) / batch_size
                if num_batch > 1:
                    r = ceil(num_batch)*batch_size - len(labels)
                    song = torch.concat((song, song[:, :r*cfg.CHUNK_LEN]), dim=1)
                    labels = torch.concat((labels, labels[:r]))

                song_dataset = SongDataset(song, labels)
                song_loader = DataLoader(
                    song_dataset, batch_size=batch_size, shuffle=True)

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
        score_avg, loss_avg = validate(model, val_loader, criterion)
        checkpoint = {'state_dict': model.state_dict(),
                      'best_score': best_score}

        # save the best model
        if score_avg > best_score:
            best_score = score_avg
            checkpoint['best_score'] = best_score
            torch.save(checkpoint, os.path.join(exp_dir, args.model+'_best.pt'))
        # save the latest model
        torch.save(checkpoint, os.path.join(exp_dir, args.model+'_last.pt'))

        logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_loss={:.5f}\t score={:.3f}\t best_score{:.3f}'.format(
            epoch, n_epochs, loss_sum/batch_sum, loss_avg, score_avg, best_score))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--model', type=str,
                        default='unsup_embedding', choices=['unsup_embedding'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrained', type=str,
                        help='the path of pre-trained model')

    args = parser.parse_args()
    train(args)
