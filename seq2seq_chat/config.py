# encoding=utf-8
import json
import os
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

xhj_corpus_url = 'https://github.com/candlewill/Dialog_Corpus/raw/master/xiaohuangji50w_nofenci.conv.zip'
ptt_corpus_url = 'https://github.com/zake7749/Gossiping-Chinese-Corpus/raw/master/data/Gossiping-QA-Dataset.txt'

xhj_corpus_loc = 'data/xiaohuangji50w_nofenci.conv'
ptt_corpus_loc = 'data/Gossiping-QA-Dataset.txt'
wordmap_loc = 'data/WORDMAP.json'
samples_loc = 'data/samples.json'

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 100
save_every = 100
workers = 1
max_len = 10  # Maximum sentence length to consider
min_word_freq = 3  # Minimum word count threshold for trimming
save_dir = 'models'

# Configure models
model_name = 'cb_model'
attn_model = 'general'
start_epoch = 0
epochs = 502
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 1024

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000

num_samples = 298154    #小黄鸡语料大小
num_training_samples = 290000

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3


class Voc:
    def __init__(self, filename):
        word_map = json.load(open(filename, 'r'))
        self.word2index = word_map
        self.index2word = {v: k for k, v in word_map.items()}
        self.num_words = len(word_map)


if os.path.isfile(wordmap_loc):
    voc = Voc(wordmap_loc)
