from keras import backend as K
from keras.engine.topology import Layer
# from keras import layers
import numpy as np
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import  *
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
        '''
	self.W_y = self.add_weight(shape = (self.output_dim, self.output_dim),
                                      initializer = 'uniform',
                                      trainable = True)
        '''
        self.W_h = self.add_weight(shape = (self.output_dim, self.output_dim),
                                      initializer = 'uniform',
                                      trainable = True)
       
        self.W_p = self.add_weight(shape = (self.output_dim, self.output_dim),
                                      initializer = 'uniform',
                                      trainable = True)
        self.W_x = self.add_weight(shape = (self.output_dim, self.output_dim),
                                      initializer = 'uniform',
                                      trainable = True)

        '''
        self.w = self.add_weight(shape = (self.output_dim,),
                                      initializer = 'uniform',
                                      trainable = True)
        '''

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def get_H_n(self, X):
    	ans = X[:, -1, :]  # get last element from time dim
    	return ans

    def get_Y(self, X, maxlen):
        print 'maxlen...', maxlen
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

    	Y = Lambda(self.get_Y, arguments = {"maxlen": self.max_len}, name = "Y")(drop_out)
        
	print ('Y shape', Y.shape)
        print ('h_n_shape', h_n.shape)

        scaled_hidden_h_n = K.dot(h_n, self.W_h)
        
        print ('scaled _h_n_shape', scaled_hidden_h_n.shape)

	wh_n_x_e = RepeatVector(self.max_len, name = "Wh_n_x_e")(scaled_hidden_h_n)

        print('wh_n_x_e shape', wh_n_x_e.shape)
        
        #k*k l*k l*k
        M_input  = TimeDistributed(Dense(self.output_dim))(Y)
        
        
        
	M = Activation('tanh')( M_input  + wh_n_x_e)
	
	print ('Shape of M', M.shape)
   	#print ('shape of w', type(self.w))

	alpha_input = TimeDistributed(Dense(1))(M)
	print('alpha_input...', alpha_input.shape)
	#self.w = Reshape((self.output_dim,1) , input_shape = (self.output_dim,))(self.w)
	alpha = Activation('softmax')(alpha_input)
	
	print('alpha shape', alpha.shape)
        # alpha is 1 * l
        # Y is k * l
        r = K.batch_dot(Y , alpha, axes=[1,1])

        #scaled_r  = self.W_p * r
         
        #scaled_h_n = self.W_x * h_n
          
        #scaled_r = TimeDistributed(Dense(self.output_dim))(r)

	#scaled_h_n = TimeDistributed(Dense(self.output_dim))(h_n)

	print('r shape', r.shape)
	r = Reshape((self.output_dim,))(r)
	print('r shape', r.shape)
	h_star = Activation('tanh')(K.dot(r, self.W_p)  + K.dot( h_n ,self.W_x) )

	print('h_star shape...', h_star.shape)
    	return h_star

    def compute_output_shape(self, input_shape):

        return (input_shape[0], self.output_dim)
