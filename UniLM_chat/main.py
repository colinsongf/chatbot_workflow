import numpy as np
from tqdm import tqdm
import os, json, codecs
from bert4keras.bert import load_pretrained_model
from bert4keras.utils import SimpleTokenizer, load_vocab
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
from flask import Flask, request


config_path = 'seq2seq_config.json'
max_input_len = 32
max_output_len = 32

chars = json.load(open(config_path))

config_path = './chinese_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_wwm_ext_L-12_H-768_A-12/vocab.txt'


_token_dict = load_vocab(dict_path) # 读取词典
token_dict, keep_words = {}, []

for c in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[unused1]']:
    token_dict[c] = len(token_dict)
    keep_words.append(_token_dict[c])

for c in chars:
    if c in _token_dict:
        token_dict[c] = len(token_dict)
        keep_words.append(_token_dict[c])


tokenizer = SimpleTokenizer(token_dict) # 建立分词器


model = load_pretrained_model(
    config_path,
    checkpoint_path,
    seq2seq=True,
    keep_words=keep_words
)


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

model.load_weights('./best_model.weights')
model._make_predict_function()
graph = tf.get_default_graph()

def gen_sent(s, topk=2):
    """beam search解码
    每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
    """
    token_ids, segment_ids = tokenizer.encode(s[:max_input_len])
    target_ids = [[] for _ in range(topk)] # 候选答案id
    target_scores = [0] * topk # 候选答案分数

    # global graph

    for i in range(max_output_len): # 强制要求输出不超过max_output_len字
        _target_ids = [token_ids + t for t in target_ids]
        _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]

        # with graph.as_default():
        _probas = model.predict([_target_ids, _segment_ids])[:, -1, 3:] # 直接忽略[PAD], [UNK], [CLS]
        
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

app = Flask(__name__)

@app.route('/chat', methods = ['POST'])
def chat():
    input_text = request.json.get('text', ' ')

    result = {"text": gen_sent(input_text)}

    return json.dumps(result, ensure_ascii=False)

if __name__ == "__main__":

    app.run(port=8080, debug=False, host='0.0.0.0', threaded=False)

