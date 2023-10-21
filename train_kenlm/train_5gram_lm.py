import os
import pandas as pd


os.system('sudo apt -y install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev')
os.system('wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz')
os.system('mkdir kenlm/build && cd kenlm/build && cmake .. && make -j2')
os.system('chmod +x kenlm/build/bin/lmplz')



with open("../data/normalized_bn_without_punc.txt", "r", encoding='utf-8') as read_file, open("../data/normalized_bn_train_commonvoice_fleurs_openslr_openslr37_oscar_without_punc.txt", "w", encoding='utf-8') as write_file:
    for line in read_file:
        write_file.write(line + ' ')

df = pd.read_csv('../data/preprocessed_train_with_audio_length.csv')
sentences = df['sentence'].values
with open("../data/normalized_bn_train_commonvoice_fleurs_openslr_openslr37_oscar_without_punc.txt", "a", encoding='utf-8') as write_file:
    for line in sentences:
        write_file.write(line + ' ')

with open('../data/normalized_cleaned_commonvoice.txt', 'r', encoding='utf-8') as f:
    sentences1 = f.readlines()
with open('../data/normalized_cleaned_fleurs.txt', 'r', encoding='utf-8') as f:
    sentences2 = f.readlines()
with open('../data/normalized_cleaned_openslr.txt', 'r', encoding='utf-8') as f:
    sentences3 = f.readlines()
with open('../data/normalized_cleaned_openslr37.txt', 'r', encoding='utf-8') as f:
    sentences4 = f.readlines()
with open('../data/normalized_cleaned_oscar.txt', 'r', encoding='utf-8') as f:
    sentences5 = f.readlines()


with open("../data/normalized_bn_train_commonvoice_fleurs_openslr_openslr37_oscar_without_punc.txt", "a", encoding='utf-8') as write_file:
    for line in sentences1:
        write_file.write(line + ' ')
    for line in sentences2:
        write_file.write(line + ' ')
    for line in sentences3:
        write_file.write(line + ' ')
    for line in sentences4:
        write_file.write(line + ' ')
    for line in sentences5:
        write_file.write(line + ' ')

os.system('kenlm/build/bin/lmplz -o 5 --prune 2 < "../data/normalized_bn_train_commonvoice_fleurs_openslr_openslr37_oscar_without_punc.txt" > "5gram.arpa"')
with open("5gram.arpa", "r") as read_file, open("5gram_normalized_bn_train_commonvoice_fleurs_openslr_openslr37_oscar_without_punc_prune2.arpa", "w") as write_file:
    has_added_eos = False
    for line in read_file:
        if not has_added_eos and "ngram 1=" in line:
            count=line.strip().split("=")[-1]
            write_file.write(line.replace(f"{count}", f"{int(count)+1}"))
        elif not has_added_eos and "<s>" in line:
            write_file.write(line)
            write_file.write(line.replace("<s>", "</s>"))
            has_added_eos = True
        else:
            write_file.write(line)
