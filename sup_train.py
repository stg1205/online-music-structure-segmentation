from argparse import ArgumentParser
from math import ceil
from tqdm import trange
from supervised_model.mss_data import HarmonixDataset, SongDataset
from supervised_model.sup_model import UnsupEmbedding
import supervised_model.config as cfg
from pytorch_metric_learning.losses import MultiSimilarityLoss
from torch.utils.data import DataLoader
import torch
import time
import os
from utils.logger import create_logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(model, song, labels):
    '''
    validate the MSS performance using MSA algorithm
    '''
    score = 0
    return score


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
    batch_size = args.batch_size
    lr = args.lr
    n_epochs = args.n_epochs
    # models
    if args.model == 'unsup_embedding':
        model = UnsupEmbedding((cfg.BIN, cfg.CHUNK_LEN))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = MultiSimilarityLoss()
    dataset = HarmonixDataset()

    dataset_loader = DataLoader(dataset, shuffle=True)

    best_score = 0
    exp_dir, logger = experiment_setup(args)
    for epoch in range(n_epochs):
        model.train()
        iterator = iter(dataset_loader)
        with trange(len(dataset_loader)) as t:
            for _ in t:
                song, labels = next(iterator)
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
                    #print(examples.shape)
                    example_labels = example_labels.to(device)
                    #print(example_labels.shape)
                    embeddings = model(examples)
                    loss = criterion(embeddings, example_labels)

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                t.set_description('Epoch:[{}/{}], loss={:.5f}'.format(epoch, n_epochs, loss.item()))
        # validate TODO
        score = validate(model, song, labels)
        checkpoint = {'state_dict': model.state_dict(),
                      'best_score': best_score}

        # save the best model
        if score > best_score:
            best_score = score
            checkpoint['best_score'] = best_score
            torch.save(checkpoint, os.path.join(exp_dir, args.model+'_best.pt'))
        # save the latest model
        torch.save(checkpoint, os.path.join(exp_dir, args.model+'_last.pt'))

        logger.info(
            'Epoch:[{}/{}]\t loss={:.5f}\t score={:.3f}'.format(epoch, n_epochs, loss.item(), score))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--model', type=str,
                        default='unsup_embedding', choices=['unsup_embedding'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--pretrained', type=str)

    args = parser.parse_args()
    train(args)
