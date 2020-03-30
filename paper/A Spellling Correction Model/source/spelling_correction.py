import tensorflow as tf
from tensorflow.keras.layers import *
class EncoderLayer(tf.keras.layers.layer):
    def __init__(self, dimension, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.lstmlayer = lstm(dimension, activation = 'relu', return_sequences = True)
        self.bi_direction_lstm = Bidirectional(lstm(dimension, activation = 'relu', return_sequences = True), merge_mode = 'concat')
        self.layernorm = LayerNormalization(epsilon = 1e-6)
        self.dropout = Dropout(dropout_rate)
        self.add = Add()

        def call(self, x, training):
            lstm_output = self.lstmlayer(x)
            bi_lstm_output = self.bi_direction_lstm(lstm_output)
            bi_lstm_output = self.LayerNormalization(bi_lstm_output)
            bi_lstm_output = self.dropout(bi_lstm_output, training = training)
            out = self.add(x + bi_lstm_output)
            return out

            class Encoder(tf.keras.layers.layer):
                def __init__(self, input_vocab_size, embedding_size, dimension, dropout_rate, layer_num, embedding_matrix = None):
                    super(Encoder, self).__init__()
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

                        def call(self, x, trianing):
                            embed_output = self.embedding(x)
                            drop_output = self.dropout(embed_output, training = training)
                            enc_output = self.enc_layeres(drop_output)
                            out = self.projection_layer(enc_output)
                            return out

                            def scaled_dot_product_attention(q, k, v, mask):
                                matmul_qk = tf.matmul(q, k, transpose_b = True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)# scailing을 위한 차원 계산

    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)# scailing
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis = -1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

        def scaled_additive_attention(q, k, v, mask):
            
            add_qk = tf.add(q + k)

            class MultiHeadAttention(tf.keras.layers.layer):
                def __init__(self, num_heads, d_model):
                    super(MultiHeadAttention, self).__init__()
                    self.num_heads = num_heads
                    self.depth = d_model

                    self.wq = Dense(d_model)
                    self.wk = Dense(d_model)
                    self.dense = Dense(d_model/num_heads)

                    def additive_attention(q,k):
                        add_qk = tf.nn.tanh(q + k)
                        add_output = dense(add_qk)
                        return add_output

                        def split_heads(self, x, batch_size):
                            x = tf.reshape(x, shape = (batch_size, -1, self.num_heads, self.depth))
                            x = x.transpose(x, perm = [0, 2, 1, 3])
                            return x

                            def call(self, k, q):
                                batch_size = q.shape[0]

                                q = self.wq(q)
                                k = self.wk(k)

                                q = self.split_heads(q, x, batch_size)
                                k = self.split_heads(k, x, batch_size)

                                attention_output = self.additive_attention(q, k)
                                attention_output = tf.transpose(attention_output, perm = [0, 2, 1, 3])
                                concat_attention = tf.reshape(attention_output, (batch_size, -1, self.d_model))

                                output = self.dense(concat_attention)



