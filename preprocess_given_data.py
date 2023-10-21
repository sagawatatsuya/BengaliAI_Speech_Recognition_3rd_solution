import random
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from bnunicodenormalizer import Normalizer
import re
import librosa
import sys
from utils import *

class CFG:
    SEED = 42
    N_SPLITS = 40
    INPUT_DIR = 'data/bengaliai-speech'
    SAMPLING_RATE = 16_000
seed_everything(CFG.SEED)

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\।\—]'

def remove_special_characters(text):
    new_text = re.sub(chars_to_ignore_regex, '', text)
    return new_text

def read_audio(mp3_path, target_sr=16000):
    audio, sr = librosa.load(mp3_path, sr=32000)
    return audio

def get_audio_length(row):
    audio = read_audio(row['path'])
    row['audio_length'] = len(audio)/32000
    return row

def read_data(mode):
    '''
    sentence normalization
    '''

    df = pd.read_csv(CFG.INPUT_DIR+'/train.csv')
    ids = df['id'].values
    targets = df['sentence'].values

    paths = CFG.INPUT_DIR + f'/{mode}_mp3s/' + ids + '.mp3'

    bnorm = Normalizer()
    def normalize(sen):
        _words = [bnorm(word)['normalized']  for word in sen.split()]
        return " ".join([word for word in _words if word is not None])

    targets = [remove_special_characters(target) for target in targets]

    targets = np.frompyfunc(normalize, 1, 1)(targets)

    df['path'] = paths
    df['sentence'] = targets

    df = df.apply(get_audio_length, axis=1)

    kf = StratifiedGroupKFold(n_splits = CFG.N_SPLITS, shuffle = True, random_state = CFG.SEED)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(df, df['split'], df['sentence'])):
        df.loc[val_idx, 'fold'] = fold

    df.to_csv('data/preprocessed_train_with_audio_length.csv')

read_data(mode = 'train')