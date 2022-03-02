import librosa
import numpy as np
import scipy.ndimage
import sklearn
from . import config as cfg
import msaf
import msaf.algorithms.scluster.main2 as scl
import msaf.utils as U
import pandas as pd

def idx_to_time(boundary_ids, mode='last'):
    if mode == 'last':
        return (boundary_ids * cfg.eval_hop_size + cfg.CHUNK_LEN - cfg.time_lag_len) * cfg.BIN_TIME_LEN
    elif mode == 'center':
        return (boundary_ids * cfg.eval_hop_size + cfg.CHUNK_LEN/2) * cfg.BIN_TIME_LEN

def time_to_interval(times):
    return np.array(list(zip(times[:-1], times[1:])))

def read_ref_file(fp):
    label_df = pd.read_csv(fp, sep=' ', header=None, names=['time', 'label'])
    times, labels = np.array(label_df['time'], dtype='float32'), label_df['label']

    labels = labels.str.replace('\d+', '')
    labels = labels.str.lower()
    labels = labels.str.strip()
    
    return times, np.array(labels[:-1])

def scluster(embeddings, ref_times, ref_labels):
    '''
    reference: 
    https://librosa.org/doc/0.9.1/auto_examples/plot_segmentation.html?highlight=feature%20sync
    Brian McFee, Daniel P.W. Ellis. “Analyzing song structure with spectral clustering”, 
    15th International Society for Music Information Retrieval Conference, 2014.

    Both S_rep and S_loc use embeddings as the input feature

    input:
    `embeddings`: (n_feature, time)
    `ref_times`: (m+1,)
    `ref_labels`: (m,)
    output:
    `res`: Contains the results of all the evaluations for the given file.
        Keys are the following:
            track_id: Name of the track
            HitRate_3F: F-measure of hit rate at 3 seconds
            HitRate_3P: Precision of hit rate at 3 seconds
            HitRate_3R: Recall of hit rate at 3 seconds
            HitRate_0.5F: F-measure of hit rate at 0.5 seconds
            HitRate_0.5P: Precision of hit rate at 0.5 seconds
            HitRate_0.5R: Recall of hit rate at 0.5 seconds
            HitRate_w3F: F-measure of hit rate at 3 seconds weighted
            HitRate_w0.5F: F-measure of hit rate at 0.5 seconds weighted
            HitRate_wt3F: F-measure of hit rate at 3 seconds weighted and
                            trimmed
            HitRate_wt0.5F: F-measure of hit rate at 0.5 seconds weighted
                            and trimmed
            HitRate_t3F: F-measure of hit rate at 3 seconds (trimmed)
            HitRate_t3P: Precision of hit rate at 3 seconds (trimmed)
            HitRate_t3F: Recall of hit rate at 3 seconds (trimmed)
            HitRate_t0.5F: F-measure of hit rate at 0.5 seconds (trimmed)
            HitRate_t0.5P: Precision of hit rate at 0.5 seconds (trimmed)
            HitRate_t0.5R: Recall of hit rate at 0.5 seconds (trimmed)
            DevA2E: Median deviation of annotation to estimation
            DevE2A: Median deviation of estimation to annotation
            D: Information gain
            PWF: F-measure of pair-wise frame clustering
            PWP: Precision of pair-wise frame clustering
            PWR: Recall of pair-wise frame clustering
            Sf: F-measure normalized entropy score
            So: Oversegmentation normalized entropy score
            Su: Undersegmentation normalized entropy score
    '''
    est_idxs, est_labels, _ = scl.do_segmentation(embeddings, embeddings, cfg.scluster_config)
    est_idxs, est_labels = U.remove_empty_segments(est_idxs, est_labels)
    assert len(est_idxs) - 1 == len(est_labels), "Number of boundaries " \
                "(%d) and number of labels(%d) don't match" % (len(est_idxs),
                                                            len(est_labels))
    # Make sure the indices are integers
    est_idxs = np.asarray(est_idxs, dtype=int)
    est_times = idx_to_time(est_idxs, mode='last')
    est_intervals = time_to_interval(est_times)

    ref_intervals = time_to_interval(ref_times)
    res = msaf.eval.compute_results(ref_intervals, est_intervals, ref_labels, est_labels,
                                bins=251, est_file='fwaefwefwefweafewafweaf')
    
    return res