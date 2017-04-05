from keras import backend as K
from keras.engine.topology import Layer
# from keras import layers
import numpy as np
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Merge
from keras.layers.merge import Add
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.layers.core import RepeatVector
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text


class AttentionLayer(Layer):

    def __init__(self, max_len, max_features, output_dim, **kwargs):
        self.output_dim = output_dim
	self.max_len = max_len
        self.emb = output_dim
        self.max_features = max_features
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        self.W_y = self.add_weight(shape = (self.output_dim, self.output_dim),
                                      initializer = 'uniform',
                                      trainable = True)
        self.W_h = self.add_weight(shape = (self.output_dim, self.output_dim),
                                      initializer = 'uniform',
                                      trainable = True)
        self.W_p = self.add_weight(shape = (self.output_dim, self.output_dim),
                                      initializer = 'uniform',
                                      trainable = True)
        self.W_x = self.add_weight(shape = (self.output_dim, self.output_dim),
                                      initializer = 'uniform',
                                      trainable = True)


        self.w = self.add_weight(shape = (self.output_dim,),
                                      initializer = 'uniform',
                                      trainable = True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def get_H_n(self, X):
    	ans = X[:, -1, :]  # get last element from time dim
    	return ans

    def get_Y(self, X, maxlen):
	ans = X[:, :maxlen, :]
    	return ans

    def call(self, main_input):

    	drop_out = Dropout(0.1, name = 'dropout')(main_input)

    	lstm_fwd = LSTM(self.output_dim, return_sequences = True, name = 'lstm_fwd')(drop_out)

        # lstm_bwd = LSTM(self.output_dim, return_sequences = True, go_backwards = True, name = 'lstm_bwd')(drop_out)

    	# bilstm = Merge([lstm_fwd, lstm_bwd], name = 'bilstm', mode = 'sum')

    	drop_out = Dropout(0.1)(lstm_fwd)

        # h_n is 1 * k
    	h_n = Lambda(self.get_H_n, output_shape = (1, self.output_dim), name = "h_n")(drop_out)

    	Y = Lambda(self.get_Y, arguments = {"maxlen": self.max_len}, name = "Y", output_shape = (self.max_len, self.output_dim))(drop_out)

	wh_n_x_e = RepeatVector(self.max_len, name = "Wh_n_x_e")(self.W_h * K.transpose(h_n))


	M = Activation('tanh')(self.W_y * K.transpose(Y) + wh_n_x_e)

	alpha = Activation('softmax')(K.dot(K.transpose(self.w), M))

        # alpha is 1 * l
        # Y is k * l
        r = K.dot(Y, K.transpose(alpha))

	h_star = Activation('tanh')(self.W_p * r + self.W_x * K.transpose(h_n))


    	return h_star

    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.output_dim)
