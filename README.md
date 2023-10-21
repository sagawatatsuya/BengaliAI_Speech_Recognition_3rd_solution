# BengaliAI Speech Recognition 3rd place solution
We won 3rd place at the kaggle competition "BengaliAI Speech Recognition", and this is an explanation of our solution. Here, I mainly explain about how to train CTC and LM.

## 1. download dataset at following links and put them under data directory.

[competition data](https://www.kaggle.com/competitions/bengaliai-speech/data)  
[IndicCorp v2's bengali text data](https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/bn.txt
)  
[common voice's bengali audio](https://www.kaggle.com/datasets/umongsain/common-voice-13-bengali-normalized)  
[fleurs](https://huggingface.co/datasets/google/fleurs/viewer/bn_in/train)  
[openslr53](https://www.openslr.org/53/)  
[openslr37(bn_bd)](https://www.openslr.org/resources/37/bn_bd.zip)  
[openslr37(bn_in)](https://www.openslr.org/resources/37/bn_in.zip)  
[oscar](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201/viewer/bn/train?row=35)


## 2. preprocess competition csv file
```
python preprocess_given_data.py
```


## 3. train CTC model
We fine-tuned [ai4bharat/indicwav2vec_v1_bengali](https://huggingface.co/ai4bharat/indicwav2vec_v1_bengali) with competition data.  
The training process is as follows:  
1. fine-tune ai4bharat/indicwav2vec_v1_bengali with split='valid' data
```
cd train_CTC
python stage1.py
```
2. calculate WER of split='train' data with stage1 model
```
python calculate_wer_bs1.py
```
3. fine-tune ai4bharat/indicwav2vec_v1_bengali with split='valid' data and split='train' data whose WER is lower than 0.75
```
python stage2.py
```


## 4. train kenlm
1. preprocess external text data
```
python preprocess_external_data.py
```
2. train 5gram lm
```
cd train_kenlm
python train_5gram_lm.py
```


## 5. train punctuation model
please refere to https://github.com/espritmirai/bengali-punctuation-model


## 6. inference
please refere to https://www.kaggle.com/code/takuji/3rd-place-solution
