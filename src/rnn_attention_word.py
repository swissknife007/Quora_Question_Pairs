
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
# from tensorflow.contrib.keras import utils
from keras.utils import np_utils
import numpy as np
from data_helpers import fitData
import argparse
import sys
import time
import random
from sklearn.preprocessing.data import OneHotEncoder

max_len = 40

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 512
display_step = 10

# Network Parameters
n_input = 300
n_steps = max_len  # timesteps
n_hidden = 100  # hidden layer num of features
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2 * n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def get_params():
    parser = argparse.ArgumentParser(description = 'Short sample app')
    parser.add_argument('-lstm', action = "store", default = 300, dest = "lstm_units", type = int)
    parser.add_argument('-epochs', action = "store", default = 20, dest = "epochs", type = int)
    parser.add_argument('-batch', action = "store", default = 512, dest = "batch_size", type = int)
    parser.add_argument('-emb', action = "store", default = 300, dest = "emb", type = int)
    parser.add_argument('-maxlen', action = "store", default = 40, dest = "maxlen", type = int)
    parser.add_argument('-maxfeat', action = "store", default = 2000000, dest = "max_features", type = int)
    parser.add_argument('-classes', action = "store", default = 2, dest = "num_classes", type = int)
    parser.add_argument('-lr', action = "store", default = 0.001, dest = "lr", type = float)
    parser.add_argument('-verbose', action = "store", default = False, dest = "verbose", type = bool)
#     parser.add_argument('-train', action="store", default="train_all.txt", dest="train")
#     parser.add_argument('-test', action="store", default="test_all.txt", dest="test")
#     parser.add_argument('-dev', action="store", default="dev.txt", dest="dev")

    opts = parser.parse_args(sys.argv[1:])
    print ("lstm_units", opts.lstm_units)
    print ("epochs", opts.epochs)
    print ("batch_size", opts.batch_size)
    print ("emb", opts.emb)
    print ("maxlen", opts.maxlen)
    print ("max_features", opts.max_features)
    return opts

class AttentionModel:
    def __init__(self, opts, sess, MAXLEN, vocab, batch_size = 512):
        self.dim = 300
        self.sess = sess
        self.h_dim = opts.lstm_units
        self.batch_size = batch_size
        self.vocab_size = len(vocab)
        self.MAXLEN = MAXLEN

    def build_model(self):

        self.x = tf.placeholder(tf.int32, [self.batch_size, self.MAXLEN], name = "premise")
        self.x_length = tf.placeholder(tf.int32, [self.batch_size], name = "premise_len")
        self.y = tf.placeholder(tf.int32, [self.batch_size, self.MAXLEN], name = "hypothesis")
        self.y_length = tf.placeholder(tf.int32, [self.batch_size], name = "hyp_len")
        self.target = tf.placeholder(tf.float32, [self.batch_size, 2], name = "label")  # change this to int32 and it breaks.

        # DO NOT DO THIS
        # self.batch_size = tf.shape(self.x)[0]  # batch size
        # self.x_length = tf.shape(self.x)[1]  # batch size
        # print self.batch_size,self.x_length


        self.W = tf.Variable(tf.constant(0.0, shape = [self.vocab_size, self.dim]),
                        trainable = True, name = "W")

        self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.dim], name = 'emb_matrix')
        self.embed_matrix = self.W.assign(self.embedding_placeholder)

        self.x_emb = tf.nn.embedding_lookup(self.embed_matrix, self.x)
        self.y_emb = tf.nn.embedding_lookup(self.embed_matrix, self.y)

        # print self.x_emb, self.y_emb

        with tf.variable_scope("encode_q1"):
            self.fwd_lstm = rnn.BasicLSTMCell(self.h_dim, state_is_tuple = True)
            self.x_output, self.x_state = tf.nn.dynamic_rnn(cell = self.fwd_lstm, inputs = self.x_emb, dtype = tf.float32)
            # self.x_output, self.x_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fwd_lstm,cell_bw=self.bwd_lstm,inputs=self.x_emb,dtype=tf.float32)
            # print self.x_output
            # print self.x_state

        # print tf.shape(self.x)
        with tf.variable_scope("encode_q2"):
            self.fwd_lstm = rnn.BasicLSTMCell(self.h_dim, state_is_tuple = True)
            self.y_output, self.y_state = tf.nn.dynamic_rnn(cell = self.fwd_lstm, inputs = self.y_emb,
                                                            initial_state = self.x_state, dtype = tf.float32)
            # print self.y_output
            # print self.y_state

        self.Y = self.x_output  # its length must be x_length
	
	self.H = self.y_output # the set of hidden states from Q2
 	
	
  	self.r = tf.get_variable("r", initializer = tf.ones([self.batch_size, self.h_dim], dtype = tf.float32))
	
	self.H_transposed = tf.transpose(self.H, perm=[1, 0, 2])
	
	self.W_Y = tf.get_variable("W_Y", shape = [self.h_dim, self.h_dim])

        self.W_h = tf.get_variable("W_h", shape = [self.h_dim, self.h_dim])

        tmp1 = tf.matmul(tf.reshape(self.Y, shape = [self.batch_size * self.MAXLEN, self.h_dim]), self.W_Y,
                         name = "Wy")
        self.Wy = tf.reshape(tmp1, shape = [self.batch_size, self.MAXLEN, self.h_dim])
	
	self.W_r = tf.get_variable("W_r", shape = [self.h_dim, self.h_dim])

	self.W_x = tf.get_variable("W_x", shape = [self.h_dim, self.h_dim])
	
	self.W_p = tf.get_variable("W_p", shape = [self.h_dim, self.h_dim])
	
	self.W_t = tf.get_variable("W_t", shape = [self.h_dim, self.h_dim])
	
	
	
	tmp5 = tf.transpose(self.y_output, [1, 0, 2])

        self.h_n = tf.gather(tmp5, int(tmp5.get_shape()[0]) - 1)

        self.W_att = tf.get_variable("W_att", shape = [self.h_dim, 1])
	
  	outputs = []
  	for t in xrange(1, self.MAXLEN):

	    h_t = self.H_transposed[t:t+1,:,]
           
            h_t_reshaped = tf.reshape(h_t, shape = [self.batch_size, self.h_dim])
            
	    print ('h_t_shape', h_t_reshaped.shape)

            h_t_repeat = tf.expand_dims(h_t_reshaped, 1)

	    print('h_t_repeat', h_t_repeat.shape)

            pattern = tf.stack([1, self.MAXLEN, 1])
		
            h_t_repeat = tf.tile(h_t_repeat, pattern)
            
	    tmp2 = tf.matmul(tf.reshape(h_t_repeat, shape = [self.batch_size * self.MAXLEN, self.h_dim]), self.W_h)
	
	    Wh_t = tf.reshape(tmp2, shape = [self.batch_size, self.MAXLEN, self.h_dim], name = "Wh_t")
            
            r_repeat = tf.expand_dims(self.r, 1)

            r_repeat = tf.tile(r_repeat, pattern)
            
	    tmp3 = tf.matmul(tf.reshape(r_repeat, shape = [self.batch_size * self.MAXLEN, self.h_dim]), self.W_r)
	
	    Wr_r = tf.reshape(tmp3, shape = [self.batch_size, self.MAXLEN, self.h_dim], name = "Wr_r")
	    
	    M_input = tf.add(self.Wy, tf.add(Wh_t, Wr_r))
 
	    M_t = tf.tanh(M_input)
           
	    tmp4 = tf.matmul(tf.reshape(M_t, shape = [self.batch_size * self.MAXLEN, self.h_dim]), self.W_att)
        
            att = tf.nn.softmax(tf.reshape(tmp4, shape = [self.batch_size, 1, self.MAXLEN], name = "att"))
	    
	    self.att = att
	
	    r_future = tf.reshape(tf.matmul(att, self.Y), shape = [self.batch_size, self.h_dim])
	
	    Wt_r = tf.matmul(self.r, self.W_t, name = "Wt_r")    	
	     	
	    self.r = tf.add(r_future, tf.tanh(Wt_r))

    	   
  

	self.Wpr = tf.matmul(self.r, self.W_p, name = "Wpr") 

        self.Wxhn = tf.matmul(self.h_n, self.W_x, name = "Wxhn")

        self.hstar = tf.tanh(tf.add(self.Wpr, self.Wxhn), name = "hstar")
	


        self.W_pred = tf.get_variable("W_pred", shape = [self.h_dim, 2])
        self.pred = tf.nn.softmax(tf.matmul(self.hstar, self.W_pred), name = "pred_layer")
        # print "pred",self.pred,"target",self.target
        correct = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.target, 1))
        self.acc = tf.reduce_mean(tf.cast(correct, "float"), name = "accuracy")
        # self.H_n = self.last_relevant(self.en_output)
        self.loss = -tf.reduce_sum(self.target * tf.log(self.pred), name = "loss")
        # print self.loss
        self.optimizer = tf.train.AdamOptimizer()
        self.optim = self.optimizer.minimize(self.loss, var_list = tf.trainable_variables())
        _ = tf.summary.scalar("loss", self.loss)

    def train(self, \
              xdata, ydata, zdata, x_lengths, y_lengths, \
              xxdata, yydata, zzdata, xx_lengths, yy_lengths, \
              glove_matrix, MAXITER):
        merged_sum = tf.summary.merge_all()
        # writer = tf.train.SummaryWriter("./logs/%s" % "modeldir", self.sess.graph_def)
        tf.initialize_all_variables().run()
        start_time = time.time()
        for ITER in range(MAXITER):
            # xdata, ydata, zdata, x_lengths, y_lengths = joint_shuffle(xdata, ydata, zdata, x_lengths, y_lengths)
            for i in xrange(0, len(xdata), self.batch_size):
                x, y, z, xlen, ylen = xdata[i:i + self.batch_size], \
                                ydata[i:i + self.batch_size], \
                                zdata[i:i + self.batch_size], \
                                x_lengths[i:i + self.batch_size], \
                                y_lengths[i:i + self.batch_size]
                feed_dict = {self.x: x, \
                             self.y: y, \
                             self.target: z, \
                             self.x_length:xlen, \
                             self.y_length:ylen, \
                             self.embedding_placeholder:glove_matrix}
                att, _ , loss, acc, summ = self.sess.run([self.att, self.optim, self.loss, self.acc, merged_sum], feed_dict = feed_dict)
                # print "att for 0th",att[0]
                print ("loss", loss, "acc on train", acc)
            total_test_acc = []
            for i in xrange(0, len(xxdata), self.batch_size):
                x, y, z, xlen, ylen = xxdata[i:i + self.batch_size], \
                                yydata[i:i + self.batch_size], \
                                zzdata[i:i + self.batch_size], \
                                xx_lengths[i:i + self.batch_size], \
                                yy_lengths[i:i + self.batch_size]
                tfeed_dict = {self.x: x, \
                              self.y: y, \
                              self.target: z, \
                              self.x_length:xlen, \
                              self.y_length:ylen, \
                              self.embedding_placeholder:glove_matrix}
                att, _ , test_loss, test_acc, summ = self.sess.run([self.att, self.optim, self.loss, self.acc, merged_sum], feed_dict = tfeed_dict)
                total_test_acc.append(test_acc)
            print("acc on test", np.mean(total_test_acc))
        # for x, y, z in zip(xdata, ydata, zdata):
            # print x, y, z
            # feeddict = {self.x: x, self.y: y, self.target: z, self.x_length:x_lengths, self.y_length:y_lengths}
            # self.sess.run([self.optim, self.loss, merged_sum],feed_dict=feeddict);
        elapsed_time = time.time() - start_time
        print("total time", elapsed_time)


def joint_shuffle(xdata, ydata, zdata, x_lengths, y_lengths):
    tmp = list(zip(xdata, ydata, zdata, x_lengths, y_lengths))
    random.shuffle(tmp)
    xdata, ydata, zdata, x_lengths, y_lengths = zip(*tmp)
    return xdata, ydata, zdata, x_lengths, y_lengths

if __name__ == "__main__":

    options = get_params()

#     train = [l.strip().split('\t') for l in open(options.train)]
#     dev = [l.strip().split('\t') for l in open(options.dev)]
#     test = [l.strip().split('\t') for l in open(options.test)]
#     vocab = get_vocab(train)

    X_train, Y_train, X_dev, Y_dev, X_test, Y_test, Z_train, Z_dev, Z_test, vocab_dict, glove_matrix = fitData()

#     X_train, Y_train, Z_train = load_data(train, vocab)
#     X_dev, Y_dev, Z_dev = load_data(dev, vocab)
#     X_test, Y_test, Z_test = load_data(test, vocab)
    # print Z_train[1]
    # sys.exit()

    X_train_lengths = [len(x) for x in X_train]
    X_dev_lengths = np.asarray([len(x) for x in X_dev]).reshape(len(X_dev))
    X_test_lengths = np.asarray([len(x) for x in X_test]).reshape(len(X_test))
    # print len(X_test_lengths)

    Y_train_lengths = np.asarray([len(x) for x in Y_train]).reshape(len(Y_train))
    Y_dev_lengths = np.asarray([len(x) for x in Y_dev]).reshape(len(Y_dev))
    Y_test_lengths = np.asarray([len(x) for x in Y_test]).reshape(len(Y_test))
    # print len(Y_test_lengths)

    Z_train = np_utils.to_categorical(Z_train, nb_classes = options.num_classes)
    Z_dev = np_utils.to_categorical(Z_dev, nb_classes = options.num_classes)
    Z_test = np_utils.to_categorical(Z_test, nb_classes = options.num_classes)

    # print Z_train[0]

    MAXLEN = options.maxlen
    MAXITER = 1000
#     X_train = pad_sequences(X_train, maxlen=XMAXLEN, value=vocab["unk"], padding='post') ## NO NEED TO GO TO NUMPY , CAN GIVE LIST OF PADDED LIST
#     X_dev = pad_sequences(X_dev, maxlen=XMAXLEN, value=vocab["unk"], padding='post')
#     X_test = pad_sequences(X_test, maxlen=XMAXLEN, value=vocab["unk"], padding='post')
#     Y_train = pad_sequences(Y_train, maxlen=YMAXLEN, value=vocab["unk"], padding='post')
#     Y_dev = pad_sequences(Y_dev, maxlen=YMAXLEN, value=vocab["unk"], padding='post')
#     Y_test = pad_sequences(Y_test, maxlen=YMAXLEN, value=vocab["unk"], padding='post')
    # print (X_test.shape, X_test_lengths.shape)

#     vocab = get_vocab(train)

    with tf.Session() as sess:
        model = AttentionModel(options, sess, MAXLEN, vocab_dict, batch_size = 512)
        model.build_model()
        model.train(X_train, Y_train, Z_train, X_train_lengths, Y_train_lengths, \
                    X_test, Y_test, Z_test, X_test_lengths, Y_test_lengths, \
                    glove_matrix, MAXITER)
