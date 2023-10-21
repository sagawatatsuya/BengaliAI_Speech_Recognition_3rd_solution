import pandas as pd
import librosa
import datasets
from bnunicodenormalizer import Normalizer
import chardet
import os
import huggingface_hub
import re

bnorm = Normalizer()
def normalize(sen):
    _words = [bnorm(word)['normalized']  for word in sen.split()]
    return " ".join([word for word in _words if word is not None])


# preprocess IndicCorp v2 bengali dataset
with open("data/indiccorp-v2-bengali/bn.txt", "r", encoding='utf-8') as read_file, open("data/normalized_bn_with_punc.txt", "w", encoding='utf-8') as write_file:
    for line in read_file:
        new_line = normalize(line)
        write_file.write(new_line + '\n')

# preprocess fleurs dataset
ds = datasets.load_dataset('google/fleurs', 'bn_in')
ds = ds.remove_columns(['id', 'num_samples', 'audio', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
ds_train = ds['train']
ds_test = ds['test']
ds_val = ds['validation']
ds = datasets.concatenate_datasets([ds_train, ds_test, ds_val])
df = ds.to_pandas()
with open("data/normalized_fleurs.txt", "w", encoding='utf-8') as write_file:
    for line in df['transcription'].values:
        write_file.write(line + '\n')


# preprocess openslr53
# data is downloaded from https://www.openslr.org/53/
df = pd.read_csv('data/openslr53/utt_spk_text.tsv', sep='\t', names=['id', 'speaker', 'sequence'])
with open("data/normalized_openslr.txt", "w", encoding='utf-8') as write_file:
    for line in df['sequence'].values:
        new_line = normalize(line)
        write_file.write(new_line + '\n')


# preprocess commonvoice
# data is downloaded from https://www.kaggle.com/datasets/umongsain/common-voice-13-bengali-normalized
train = pd.read_csv('data/common-voice-13-bengali-normalized/train.tsv', sep='\t')
test = pd.read_csv('data/common-voice-13-bengali-normalized/test.tsv', sep='\t')
dev = pd.read_csv('data/common-voice-13-bengali-normalized/dev.tsv', sep='\t')
other = pd.read_csv('data/common-voice-13-bengali-normalized/other.tsv', sep='\t')

all = pd.concat([train, test, dev, other])
with open("data/normalized_commonvoice.txt", "w", encoding='utf-8') as write_file:
    for line in all['sentence'].values:
        new_line = normalize(line)
        write_file.write(new_line + '\n')


# preprocess oscar dataset
# data is downloaded from huggingface hub. Before downloading, you need to login to huggingface hub and accept the terms and conditions
my_token = os.getenv("HUGGINGFACE_TOKEN")
huggingface_hub.login(my_token)
dataset = datasets.load_dataset('oscar-corpus/OSCAR-2201', 'bn', split='train', use_auth_token=True)
with open('data/oscar.txt', 'w', encoding='utf-8') as f:
  for text in dataset['text']:
    # if there is alphabet characters in the text, skip it
    if not bool(re.search(r'[a-zA-Z]', text)):
      f.write(text + '\n') 

with open("data/oscar.txt", "r", encoding='utf-8') as read_file, open("data/normalized_oscar.txt", "w", encoding='utf-8') as write_file:
    for line in read_file:
        new_line = normalize(line)
        write_file.write(new_line + '\n')


# preprocess openslr37
# data is downloaded from https://www.openslr.org/53/
with open("data/normalized_openslr37.txt", "w", encoding='utf-8') as write_file:
    df1 = pd.read_csv('data/openslr_37/bn_bd/line_index.tsv', sep='\t', names=['path', 'sentence'])
    df2 = pd.read_csv('data/openslr_37/bn_in/line_index.tsv', sep='\t', names=['path', 'sentence'])
    for line in df1['sentence'].values:
        new_line = normalize(line)
        write_file.write(new_line + '\n')
    for line in df2['sentence'].values:
        new_line = normalize(line)
        write_file.write(new_line + '\n')



# remove punctuations
os.system("sed 's/[,?.!\-\;\:\"“%‘”�—’…–]//g' data/normalized_bn_with_punc.txt > data/normalized_bn_without_punc.txt")
os.system("sed 's/[,?.!\-\;\:\"“%‘”�—’…–]//g' data/normalized_commonvoice.txt > data/normalized_cleaned_commonvoice.txt")
os.system("sed 's/[,?.!\-\;\:\"“%‘”�—’…–]//g' data/normalized_fleurs.txt > data/normalized_cleaned_fleurs.txt")
os.system("sed 's/[,?.!\-\;\:\"“%‘”�—’…–]//g' data/normalized_openslr.txt > data/normalized_cleaned_openslr.txt")
os.system("sed 's/[,?.!\-\;\:\"“%‘”�—’…–]//g' data/normalized_oscar.txt > data/normalized_cleaned_oscar.txt")
os.system("sed 's/[,?.!\-\;\:\"“%‘”�—’…–]//g' data/normalized_openslr37.txt > data/normalized_cleaned_openslr37.txt")
