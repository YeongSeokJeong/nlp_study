import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import time
from tensorflow.keras.layers import *
from tensorflow.keras import Model
import pickle
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
        self.lstmlayer = LSTM(dimension, activation = 'relu', 
                               return_sequences = True,
                               recurrent_initializer='glorot_uniform')
        self.bi_direction_lstm = Bidirectional(LSTM(dimension, 
                                                    return_sequences = True,
                                                    recurrent_initializer='glorot_uniform'),
                                               merge_mode = 'concat')
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

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.depth = int(d_model/num_heads)
        self.d_model = d_model
        
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.v = Dense(d_model/num_heads)
        self.batch_norm = LayerNormalization()
        
    def additive_attention(self, q,k):
        batch_size = q.shape[0]
        add_qk = tf.nn.tanh(tf.math.add(q, k))
        add_qk = self.v(add_qk)
        add_qk =  tf.nn.softmax(add_qk)
        context_vector = add_qk * q
        context_vector = tf.reduce_sum(context_vector, axis = -1)
        context_vector = tf.expand_dims(context_vector,axis = -1)
        return context_vector

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
        concat_attention = tf.expand_dims(attention_output, axis = -1)
        context_vector = tf.reduce_sum(concat_attention, axis = 1)
        context_vector = tf.reshape(context_vector, (batch_size, -1, self.num_heads))
        context_vector = self.batch_norm(context_vector)
        return context_vector

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, dimension, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.lstm = LSTM(dimension, return_sequences= True)
        self.dropout = Dropout(dropout_rate)
        self.concat = Concatenate(axis = -1)
    
    def call(self, rnn_output, attention_weights, training):
        x = self.concat([rnn_output, attention_weights])
        x = self.lstm(x)
        x = self.dropout(x, training = training)
        out = tf.math.add(x, rnn_output)
        return out

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, dimension, num_layers, embedding_dim, dropout_rate, input_vocab_size, embedding_matrix = None):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        
        if embedding_matrix:
            self.embedding = Embedding(input_vocab_size, 
                                       embedding_dim,
                                       weights = [embedding_matrix],
                                      trianable = False)
        else:
            self.embedding = Embedding(input_vocab_size, embedding_dim)
        self.multihead = MultiHeadAttention(num_heads, dimension)
        self.dropout = Dropout(dropout_rate)
        self.lstm = LSTM(dimension, return_sequences = True,recurrent_initializer='glorot_uniform')
        self.dec_layers = [DecoderLayer(dimension, dropout_rate) for _ in range(num_layers)]
        self.concat = Concatenate(axis = -1)
        self.softmax_dense = Dense(input_vocab_size)
        
    def call(self, x, encoder_output, training = True):
        x = self.embedding(x)
        x = self.dropout(x, training = training)
        x = self.lstm(x)
        
        attention_weights = multihead(x, encoder_output)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, attention_weights, training)
        x = self.concat([x, attention_weights])
        out = self.softmax_dense(x)
        out1 = tf.nn.softmax(out)
        return out1, out

encoder = Encoder(vocab_size, units*2 , units, dropout_rate, 3)
decoder = Decoder(4, units, 3, embedding_dim, dropout_rate, len(output_vocab))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='auto')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def train_step(inp, targ, trainable):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output = encoder(inp, trainable)
        dec_input = tf.expand_dims([1]* BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions,bef_softmax = decoder(dec_input, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss/int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss

EPOCHS = 20
steps_per_epoch = len(train_input_tokens) // BATCH_SIZE
for epoch in range(EPOCHS):
    start = time.time()
    
    total_loss = 0
    for batch in range(len(train_input_tokens)//BATCH_SIZE):
        batch_input = train_input_tokens[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]
        batch_output = train_output_tokens[batch*BATCH_SIZE: (batch+1)*BATCH_SIZE]

        batch_loss = train_step(batch_input, batch_output, True)
        
        total_loss += batch_loss
        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.8f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    print('Epoch {} Loss {:.8f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))