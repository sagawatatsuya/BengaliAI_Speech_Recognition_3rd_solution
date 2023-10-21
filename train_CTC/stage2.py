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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from transformers import get_cosine_schedule_with_warmup

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
    stage = '2'
    audio_dir = "../data/bengaliai-speech/train_mp3s"
    model = "ai4bharat/indicwav2vec_v1_bengali"
    language_model = "arijitx/wav2vec2-xls-r-300m-bengali"
    seed = 42
    train_bs = 6
    valid_bs = 12
    lr = 2e-4
    weight_decay = 1e-5
    n_folds = 40
    epochs = 3
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

def train_fn(fold, train_loader, model, processor, optimizer, epoch, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=config.apex)
    losses = AverageMeter()
    start = end = time.time()
    for step, data in enumerate(train_loader):
        data = {k: v.to(device) for k, v in data.items()}
        batch_size = len(data)
        with torch.cuda.amp.autocast(enabled=config.apex):
            pred = model(**data)
            loss = pred.loss
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        end = time.time()
        if step % config.print_freq == 0 or step == (len(train_loader)-1):
            LOGGER.info('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'LR: {lr:.8f} '
                  .format(epoch+1, step, len(train_loader),
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          lr=optimizer.param_groups[0]['lr'],))
    return losses.avg, None

def valid_fn(valid_loader, model, processor, metric, device):
    losses = AverageMeter()
    wers = AverageMeter()
    model.eval()
    label_list = []
    pred_list = []
    start = end = time.time()
    for step, data in enumerate(valid_loader):
        data = {k: v.to(device) for k, v in data.items()}
        batch_size = len(data)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=config.apex):
                pred = model(**data)
                loss = pred.loss
                wer = compute_wer(pred, data['labels'].cpu().detach().numpy(), processor, metric)
        losses.update(loss.item(), batch_size)
        wers.update(wer, batch_size)
        end = time.time()
        if step % config.print_freq == 0 or step == (len(valid_loader)-1):
            LOGGER.info('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'WER: {wers.val:.4f}({wers.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          wers=wers,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))

    return losses.avg, wers.avg

def train_loop(fold):
    LOGGER.info(f'================== fold: {fold} training ======================')
    train_df = pd.read_csv('../data/preprocessed_train_with_audio_length.csv')
    train_df = train_df[:len(train_df)//20]
    train_df = train_df[train_df['audio_length'] < 15].reset_index(drop=True)
    train_df['path'] = train_df['id'].apply(lambda x: os.path.join(config.audio_dir, x+'.mp3'))
    train_folds = train_df[train_df['fold'] != fold].reset_index(drop=True)
    train_folds = train_folds[train_folds.split == 'valid'].reset_index(drop=True)
    print(len(train_folds))

    if True:
        train_df2 = pd.read_csv('../data/preprocessed_train_with_audio_length.csv')
        train_df2 = train_df2[:len(train_df2)//20]
        train_df2['path'] = train_df2['id'].apply(lambda x: os.path.join(config.audio_dir, x+'.mp3'))
        train_folds2 = train_df2[train_df2['fold'] != fold].reset_index(drop=True)
        train_folds2 = train_folds2[train_folds2.split == 'train'].reset_index(drop=True).sort_values('audio_length').reset_index(drop=True)
        wers = np.array(load_pickle('stage1/split_train_wers.pkl'))
        train_folds2 = train_folds2[wers<0.75].reset_index(drop=True)
        train_folds = pd.concat([train_folds, train_folds2]).reset_index(drop=True)
        train_folds = train_folds[train_folds['audio_length'] < 15].reset_index(drop=True)
        print(len(train_folds))


    valid_folds = train_df[train_df['fold'] == fold].reset_index(drop=True)
    train_folds = train_folds.sort_values('audio_length').reset_index(drop=True)
    valid_folds = valid_folds.sort_values('audio_length').reset_index(drop=True)
    
    processor = Wav2Vec2Processor.from_pretrained(config.model)
    model = Wav2Vec2ForCTC.from_pretrained(config.model,
        ctc_loss_reduction="mean",
        ignore_mismatched_sizes=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.config.ctc_zero_infinity = True
    model.to(device)
    model.freeze_feature_encoder()


    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    num_cycles = 0.5
    scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=0, num_training_steps=config.epochs, num_cycles=num_cycles
            )
    
    wer_metric = load_metric("wer")

    valid_dataset = ASRDataset(valid_folds, config, processor)

    partial_func = partial(ctc_data_collator, processor=processor)

    valid_loader = DataLoader(valid_dataset,
                             batch_size=config.valid_bs,
                             shuffle=False,
                             collate_fn=partial_func,
                             num_workers=config.num_workers, pin_memory=True, drop_last=False)

    best_score = float('inf')

    for epoch in range(config.epochs):
        start_time = time.time()

        train_folds = shuffle_df_chunk_by_chunk(train_folds, chunk_size=config.train_bs)
        train_dataset = ASRDataset(train_folds, config, processor)
        train_loader = DataLoader(train_dataset,
                                batch_size=config.train_bs,
                                shuffle=False,
                                collate_fn=partial_func,
                                num_workers=config.num_workers, pin_memory=True, drop_last=True)
        # train
        avg_loss, avg_wer = train_fn(fold, train_loader, model, processor, optimizer, epoch, device)
        scheduler.step()
        # eval
        val_loss, valid_wer = valid_fn(valid_loader, model, processor, wer_metric, device)

        elapsed = time.time() - start_time

        if best_score > valid_wer:
            best_score = valid_wer
            LOGGER.info(f'Epoch {epoch+1} - Save Best WER: {valid_wer:.4f} Model')
            torch.save({'model': model.state_dict(),},
                        OUT_PATH + f"model_stage{config.stage}.pth")

    LOGGER.info(f'[Fold{fold}] Best WER: {best_score}')
    torch.cuda.empty_cache()
    gc.collect()

fold = 0
if __name__ == '__main__':
    train_loop(fold)