import os
import gc
import sys
from pathlib import Path
import pickle
import time
import random
import math
from functools import partial

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import librosa

from datasets import load_metric

sys.path.append('../')
from utils import *

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
)
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



class config:
    stage = '1'
    audio_dir = "../data/bengaliai-speech/train_mp3s"
    model = "ai4bharat/indicwav2vec_v1_bengali"
    language_model = "arijitx/wav2vec2-xls-r-300m-bengali"
    seed = 42
    train_bs = 1
    valid_bs = 1
    lr = 2e-4
    weight_decay = 1e-5
    n_folds = 40
    epochs = 10
    apex = True
    print_freq = 1000
    num_workers = 16
    sampling_rate = 16000

try:
    os.mkdir(f'stage{config.stage}')
except:
    pass

OUT_PATH = f'stage{config.stage}/'


LOGGER = get_logger(OUT_PATH+'train')
seed_everything(config.seed)


def read_audio(mp3_path, target_sr=16000):
    audio, sr = librosa.load(mp3_path, sr=32000)
    audio_array = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio_array

class ASRDataset(Dataset):
    def __init__(self, df, config, processor, is_test=False):
        self.df = df
        self.config = config
        self.is_test = is_test
        self.processor = processor
    
    def __getitem__(self, idx):
        audio = read_audio(self.df.loc[idx]['path'])
        audio = self.processor(
            audio, 
            sampling_rate=self.config.sampling_rate
        ).input_values[0]
        
        if self.is_test:
            return {'audio': audio, 'label': -1}
        else:
            with self.processor.as_target_processor():
                labels = self.processor(self.df.loc[idx]['sentence']).input_ids
            return {'audio': audio, 'label': labels}
        
    def __len__(self):
        return len(self.df)
    
def ctc_data_collator(batch, processor):
    input_features = [{"input_values": sample["audio"]} for sample in batch]
    label_features = [{"input_ids": sample["label"]} for sample in batch]
    batch = processor.pad(
        input_features,
        padding=True,
        return_tensors="pt",
    )
    with processor.as_target_processor():
        labels_batch = processor.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )
        
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    batch["labels"] = labels
    return batch

def compute_wer(pred, labels, processor, metric):
    pred_logits = pred.logits.cpu().detach().numpy()
    pred_ids = np.argmax(pred_logits, axis=-1)

    labels_copy = labels.copy()
    labels_copy[labels_copy == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(labels_copy, group_tokens=False)

    wer = metric.compute(predictions=pred_str, references=label_str)
    return wer

def valid_fn(valid_loader, model, processor, metric, device):
    wers = []
    model.eval()
    start = end = time.time()
    for step, data in enumerate(valid_loader):
        data = {k: v.to(device) for k, v in data.items()}
        batch_size = len(data)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=config.apex):
                pred = model(**data)
                wer = compute_wer(pred, data['labels'].cpu().detach().numpy(), processor, metric)
        wers.append(wer)
        end = time.time()
    return wers

def train_loop(fold):
    train_df = pd.read_csv('../data/preprocessed_train_with_audio_length.csv')
    train_df['path'] = train_df['id'].apply(lambda x: os.path.join(config.audio_dir, x+'.mp3'))
    folds = train_df
    train_folds = train_df[train_df['fold'] != fold].reset_index(drop=True)
    train_folds = train_folds[train_folds.split == 'train'].reset_index(drop=True)
    train_folds = train_folds.sort_values('audio_length').reset_index(drop=True)

    processor = Wav2Vec2Processor.from_pretrained(config.model)
    model = Wav2Vec2ForCTC.from_pretrained(config.model,
        ctc_loss_reduction="mean",
        ignore_mismatched_sizes=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.load_state_dict(torch.load(OUT_PATH + f"model_stage{config.stage}.pth", map_location=device)['model'])
    model.config.ctc_zero_infinity = True
    model.to(device)
    model.freeze_feature_encoder()

    wer_metric = load_metric("wer")

    train_dataset = ASRDataset(train_folds, config, processor)

    partial_func = partial(ctc_data_collator, processor=processor)

    train_loader = DataLoader(train_dataset,
                             batch_size=config.valid_bs,
                             shuffle=False,
                             collate_fn=partial_func,
                             num_workers=config.num_workers, pin_memory=True, drop_last=False)

    wers = valid_fn(train_loader, model, processor, wer_metric, device)

    with open(OUT_PATH + f'split_train_wers.pkl', 'wb') as f:
        pickle.dump(wers, f)

    torch.cuda.empty_cache()
    gc.collect()

fold = 0
if __name__ == '__main__':
    train_loop(fold)