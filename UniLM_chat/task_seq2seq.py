import numpy as np
from tqdm import tqdm
import os, json, codecs
from collections import Counter
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from data_gen import generate_dataset

seq2seq_config = 'seq2seq_config.json'
min_count = 8
max_input_len = 32
max_output_len = 32
batch_size = 32
steps_per_epoch = 17043
epochs = 51

train = True
#train = False

readdir = 'data'
readlist = os.listdir(readdir)

config_path = './chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'

#corpus = []
#pairs = []
"""
for file in readlist:
    filepath = os.path.join(readdir, file)
    fopen = codecs.open(filepath, 'r', 'utf-8')

    for line in fopen:
        if line == '\n':
            if len(pairs) == 2:
                corpus.append(pairs)
                pairs = []

        if line.startswith('Q:'):
            pairs.append(line[3:].strip())

        if line.startswith('A:'):
            pairs.append(line[3:].strip())

    fopen.close()
"""
corpus = generate_dataset()

print("corpus's length is: ", len(corpus))

if os.path.exists(seq2seq_config):
    chars = json.load(open(seq2seq_config))
else:
    chars = {}
    for pairs in corpus:  # 纯文本，不用分词
        for line in pairs:
            for w in line:
                chars[w] = chars.get(w, 0) + 1
    chars = [(i, j) for i, j in chars.items() if j >= min_count]
    chars = sorted(chars, key=lambda c: - c[1])
    chars = [c[0] for c in chars]
    json.dump(
        chars,
        codecs.open(seq2seq_config, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

_token_dict = load_vocab(dict_path) # 读取词典
token_dict, keep_words = {}, []

for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
    token_dict[c] = len(token_dict)
    keep_words.append(_token_dict[c])

for c in chars:
    if c in _token_dict and c not in token_dict:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])

print("token_dict length is : ", len(token_dict))
tokenizer = SimpleTokenizer(token_dict) # 建立分词器

def padding(x):
    """padding至batch内的最大长度
    """
    ml = max([len(i) for i in x])
    return np.array([i + [0] * (ml - len(i)) for i in x])

def data_generator():
    while True:
        X, S = [], []
        for a, b in corpus:
            x, s = tokenizer.encode(a[:max_input_len], b[:max_input_len])
            X.append(x)
            S.append(s)
            if len(X) == batch_size:
                X = padding(X)
                S = padding(S)
                yield [X, S], None
                X, S = [], []

model = load_pretrained_model(
    config_path,
    checkpoint_path,
    seq2seq=True,
    keep_words=keep_words
)

model.summary()

y_in = model.input[0][:, 1:] # 目标tokens
y_mask = model.input[1][:, 1:]
y = model.output[:, :-1] # 预测tokens，预测与目标错开一位

# 交叉熵作为loss，并mask掉输入部分的预测
y = model.output[:, :-1] # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_in, y)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(1e-5))
model.summary()


def gen_sent(s, topk=2):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    token_ids, segment_ids = tokenizer.encode(s[:max_input_len])
    target_ids = [[] for _ in range(topk)] # 候选答案id
    target_scores = [0] * topk # 候选答案分数
    for i in range(max_output_len): # 强制要求输出不超过max_output_len字
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
        _probas = model.predict(
            [_target_ids, _segment_ids]
        )[:, -1, 3:] # 直接忽略[PAD], [UNK], [CLS]
        _log_probas = np.log(_probas + 1e-6) # 取对数，方便计算
        _topk_arg = _log_probas.argsort(axis=1)[:, -topk:] # 每一项选出topk
        _candidate_ids, _candidate_scores = [], []
        for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
            if i == 0 and j > 1:
                continue
            for k in _topk_arg[j]:
                _candidate_ids.append(ids + [k + 3])
                _candidate_scores.append(sco + _log_probas[j][k])
        _topk_arg = np.argsort(_candidate_scores)[-topk:] # 从中选出新的topk
        for j, k in enumerate(_topk_arg):
            target_ids[j].append(_candidate_ids[k][-1])
            target_scores[j] = _candidate_scores[k]
        ends = [j for j, k in enumerate(target_ids) if k[-1] == 3]
        if len(ends) > 0:
            k = np.argmax([target_scores[j] for j in ends])
            return tokenizer.decode(target_ids[ends[k]])
    # 如果max_output_len字都找不到结束符，直接返回
    return tokenizer.decode(target_ids[np.argmax(target_scores)])


def just_show():
    s1 = "是个单身狗"
    s2 = "给我找师傅"
    s3 = "我想上月球"
    s4 = "冻死你个狗日的"
    for s in [s1, s2, s3, s4]:
        print('Q:', s)
        print('A:', gen_sent(s))


class Evaluate(Callback):
    def __init__(self):
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        # 演示效果
        just_show()

        if epoch%5 == 0:
            save_name = './best_model.weights_'+str(epoch)
            model.save_weights(save_name)



if train:
    evaluator = Evaluate()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1)

    model.fit_generator(
        data_generator(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')

    while True:
        input_text = input("请输入：")
        result = gen_sent(input_text)

        print("旺财：", result)
