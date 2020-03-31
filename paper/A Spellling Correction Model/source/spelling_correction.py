import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import time
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import pickle
from nltk.translate.bleu_score import sentence_bleu
import os

os.environ["CUDA_VISIBLE_DEVICES"]="2"

BATCH_SIZE = 128
embedding_dim = 300
units = 128
dropout_rate = 0.2

with open("./data/train_input_data_1.pickle", "rb") as fr:
    train_input_tokens = pickle.load(fr)

with open("./data/train_output_data_1.pickle", 'rb') as fr:
    train_output_tokens = pickle.load(fr)

with open('./data/val_input_tokens_1.pickle', 'rb') as fr:
    val_input_tokens = pickle.load(fr)

with open('./data/val_output_tokens_1.pickle', 'rb') as fr:
    val_output_tokens = pickle.load(fr) 

with open('./data/test_input_tokens_1.pickle', 'rb') as fr:
    test_input_tokens = pickle.load(fr)

with open('./data/test_output_tokens_1.pickle', 'rb') as fr:
    test_output_tokens = pickle.load(fr)

with open('./data/input_vocab_1.pickle', 'rb') as fr:
    input_vocab = pickle.load(fr)

with open('./data/output_vocab_1.pickle', 'rb') as fr:
    output_vocab = pickle.load(fr)

dic_input_vocab = {word:i for i, word in enumerate(input_vocab)}
dic_output_vocab = {word:i for i, word in enumerate(output_vocab)}
vocab_size = len(input_vocab)

input_max_len = train_input_tokens.shape[1]
output_max_len = train_output_tokens.shape[1]

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dimension, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.lstmlayer = LSTM(dimension, activation = 'relu', return_sequences = True)
        self.bi_direction_lstm = Bidirectional(LSTM(dimension, activation = 'relu', return_sequences = True), merge_mode = 'concat')
        self.layernorm = LayerNormalization(epsilon = 1e-6)
        self.dropout = Dropout(dropout_rate)

    def call(self, x, training):
        lstm_output = self.lstmlayer(x)
        bi_lstm_output = self.bi_direction_lstm(lstm_output)
        bi_lstm_output = self.layernorm(bi_lstm_output)
        bi_lstm_output = self.dropout(bi_lstm_output, training = training)
        out = tf.math.add(x = x, y = bi_lstm_output)
        return out

class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_size, dimension, dropout_rate, layer_num, embedding_matrix = None):
        super(Encoder, self).__init__()
        self.layer_num = layer_num
        if embedding_matrix : 
            self.embedding = Embedding(input_vocab_size, 
            embedding_size,
            weights = [embedding_matrix],
            trainable = False)
        else:
            self.embedding = Embedding(input_vocab_size, embedding_size)

        self.dropout = Dropout(dropout_rate)
        self.enc_layers = [EncoderLayer(dimension, dropout_rate) for _ in range(layer_num)]
        self.projection_variable = Dense(dimension, activation = 'linear')

    def call(self, x, training):
        x = self.embedding(x)
        x = self.dropout(x, training = training)
        
        for i in range(self.layer_num):
            x = self.enc_layers[i](x, training)
        out = self.projection_variable(x)
        return out

encoder = Encoder(vocab_size, units*2 , units, dropout_rate, 3)
sample_input = train_input_tokens[:BATCH_SIZE]
encoder_output = encoder(sample_input, True)
print(encoder_output.shape)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.depth = int(d_model/num_heads)
        self.d_model = d_model
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.concat_dense = Dense(1)

    def additive_attention(self, q,k):
        add_qk = tf.nn.tanh(tf.math.add(q, k))
        add_output = self.concat_dense(add_qk)
        return add_output

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, shape = (batch_size, -1, self.num_heads, self.depth))
        x = tf.transpose(x, perm = [0, 2, 1, 3])
        return x

    def call(self, k, q):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)

        q = self.split_heads(q, q.shape[0])
        k = self.split_heads(k, k.shape[0])

        attention_output = self.additive_attention(q, k)
        attention_output = tf.transpose(attention_output, perm = [0, 2, 1, 3])
        concat_attention = tf.reshape(attention_output, (batch_size, q.shape[2], -1))
        return concat_attention

class Decoder(tf.keras.layers.Layer):
    def __init__(self, dimmension, num_layers, embedding_dim, dropout_rate, input_vocab_size, embedding_matrix = None):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        
        if embedding_matrix:
            self.embedding = Embedding(input_vocab_size, 
                                       embedding_dim,
                                       weights = [embedding_matrix],
                                      trianable = False)
        else:
            self.embedding = Embedding(input_vocab_size, embedding_dim)
        self.dropout = Dropout(dropout_rate)
        self.lstm = LSTM(dimmension, return_sequences = True)
        self.dec_layers = [DecoderLayer(dimmension, dropout_rate) for _ in range(num_layers)]
        self.concat = Concatenate(axis = -1)
        self.softmax_dense = Dense(input_vocab_size, activation = 'softmax')
        
    def call(self, x, multi_head_output, training = True):
        x = self.embedding(x)
        x = self.dropout(x, training = training)
        x = self.lstm(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, multi_head_output, training)
        x = self.concat([x, multi_head_output])
        out = self.softmax_dense(x)
        return out