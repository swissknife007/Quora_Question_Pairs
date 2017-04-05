from keras import backend as K
from keras.engine.topology import Layer
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
from keras.engine.topology import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text


class AttentionLayer(Layer):

    def __init__(self, max_len,  max_features, output_dim, **kwargs):
        self.output_dim = output_dim
	self.max_len = max_len
        self.emb = output_dim
        self.max_features = max_features
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        
        # Create a trainable weight variable for this layer.
        self.W_y = self.add_weight(shape=(output_dim, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.W_h = self.add_weight(shape=(output_dim, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.W_p = self.add_weight(shape=(output_dim, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.W_x = self.add_weight(shape=(output_dim, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)

        
        self.w = self.add_weight(shape=(output_dim,),
                                      initializer='uniform',
                                      trainable=True)
	
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def get_H_n(X):
    	ans = X[:,-1,:]  # get last element from time dim
    	return ans
    
    def get_Y(X, maxlen):
	ans = X[:, :maxlen,:]
    	return ans
 
    def call(self, main_input):
	
    
    	x = Embedding(output_dim=emb, input_dim=max_features, input_length=N, name='x')(main_input)

    	drop_out = Dropout(0.1, name='dropout')(x)

    	lstm_fwd = LSTM(opts.lstm_units, return_sequences=True, name='lstm_fwd')(drop_out)

   	lstm_bwd = LSTM(opts.lstm_units, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)

    	bilstm = merge([lstm_fwd, lstm_bwd], name='bilstm', mode='sum')

    	drop_out = Dropout(0.1)(bilstm)


    	h_n = Lambda(get_H_n, output_shape=(k,), name="h_n")(drop_out)
	
    	Y = Lambda(get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, k))(drop_out)

	
	wh_n_x_e =  RepeatVector(output_dim, name="Wh_n_x_e")(self.W_h * h_n)
        
	M = Activation('tanh')(self.W_y * Y + wh_n_x_e)
    	
	alpha = Activation('softmax')(k.dot(self.w, M))
        
        r = k.dot(Y, alpha)
 	
	h_star = Activation('tanh')(self.W_p * r + self.W_x * h_n)

	
    	return h_star

    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.output_dim)
