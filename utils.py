import random
import os
import math
import time
import numpy as np
import torch
import pandas as pd
import pickle

def get_logger(filename: str):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def shuffle_df_chunk_by_chunk(df, chunk_size=1000):
    new_df = df[:len(df)//chunk_size*chunk_size].copy()
    num_sets = len(new_df) // chunk_size
    indices = list(range(num_sets))
    np.random.shuffle(indices)

    shuffled_dfs = []
    for i in indices:
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        shuffled_dfs.append(new_df.iloc[start_idx:end_idx])
        
    shuffled_df = pd.concat(shuffled_dfs)
    shuffled_df.reset_index(drop=True, inplace=True)
    return shuffled_df

def pickle_file(path, contents):
    with open(path, 'wb') as f:
        pickle.dump(contents, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        contents = pickle.load(f)
    return contents
