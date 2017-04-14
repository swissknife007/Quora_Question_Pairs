
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
# from tensorflow.contrib.keras import utils
from keras.utils import np_utils
import numpy as np
from data_helpers_kaggle import fitData, fit_test_data, write_submission_file
import argparse
import sys
import time
import random
from cnn_kaggle import get_cnn_embedding
from sklearn.preprocessing.data import OneHotEncoder
from data_augmentation import augment_data
import matplotlib.pyplot as plt
import custom_lstm

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
    parser.add_argument('-epochs', action = "store", default = 35, dest = "epochs", type = int)
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

    def __init__(self, opts, sess, MAXLEN, vocab, embedding_matrix, num_filters = 128, filter_sizes = [3, 4, 5], batch_size = 512, dropout_prob = 0.5):
        self.dim = 300
        self.sess = sess
        self.h_dim = opts.lstm_units
        self.batch_size = batch_size
        self.vocab_size = len(vocab)
        self.init_emb_matrix = embedding_matrix
        self.MAXLEN = MAXLEN
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        # self.dropout_keep_prob = dropout_prob

    def build_model(self):

        self.x = tf.placeholder(tf.int32, [self.batch_size, self.MAXLEN], name = "premise")

        self.x_length = tf.placeholder(tf.int32, [self.batch_size], name = "premise_len")

        self.y = tf.placeholder(tf.int32, [self.batch_size, self.MAXLEN], name = "hypothesis")

        self.y_length = tf.placeholder(tf.int32, [self.batch_size], name = "hyp_len")

        self.target = tf.placeholder(tf.float32, [self.batch_size, 2], name = "label")

        self.dropout_keep_prob = tf.placeholder(tf.float32, shape = [], name = "dropout_keep_prob")

	self.is_training = tf.placeholder(tf.bool, shape = [], name = "is_training")

        self.W = tf.Variable(tf.constant(0.0, shape = [self.vocab_size, self.dim]),
                        trainable = True, name = "W")

        # self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.dim], name = 'emb_matrix')

        self.embed_matrix = self.W.assign(self.init_emb_matrix)

        self.x_emb = tf.nn.embedding_lookup(self.embed_matrix, self.x)

        self.y_emb = tf.nn.embedding_lookup(self.embed_matrix, self.y)

        self.W_Y = tf.get_variable("W_Y", shape = [self.h_dim, self.h_dim])

        self.W_h = tf.get_variable("W_h", shape = [self.h_dim, self.h_dim])

        self.W_r = tf.get_variable("W_r", shape = [self.h_dim, self.h_dim])

        self.W_x = tf.get_variable("W_x", shape = [self.h_dim, self.h_dim])

        self.W_p = tf.get_variable("W_p", shape = [self.h_dim, self.h_dim])

        self.W_t = tf.get_variable("W_t", shape = [self.h_dim, self.h_dim])

        self.W_att = tf.get_variable("W_att", shape = [self.h_dim, 1])

        self.r_0 = tf.get_variable("r", initializer = tf.ones([self.batch_size, self.h_dim], dtype = tf.float32))

        with tf.variable_scope("encode_q1"):
            self.fwd_lstm = custom_lstm.BN_LSTMCell(self.h_dim, self.is_training)

            self.x_output, self.x_state = tf.nn.dynamic_rnn(cell = self.fwd_lstm, inputs = self.x_emb, dtype = tf.float32)

        with tf.variable_scope("encode_q2"):

            self.fwd_lstm = custom_lstm.BN_LSTMCell(self.h_dim, self.is_training)

            self.y_output, self.y_state = tf.nn.dynamic_rnn(cell = self.fwd_lstm, inputs = self.y_emb,
                                                            initial_state = self.x_state, dtype = tf.float32)


        self.Y = self.x_output  # its length must be x_length

        self.H = self.y_output  # the set of hidden states from Q2

        self.h_star = []

        for i in xrange(2):

            self.r = self.r_0

            self.H_transposed = tf.transpose(self.H, perm = [1, 0, 2])

            tmp5 = tf.transpose(self.y_output, [1, 0, 2])

            self.h_n = tf.gather(tmp5, int(tmp5.get_shape()[0]) - 1)

            tmp1 = tf.matmul(tf.reshape(self.Y, shape = [self.batch_size * self.MAXLEN, self.h_dim]), self.W_Y, name = "Wy")
            self.Wy = tf.reshape(tmp1, shape = [self.batch_size, self.MAXLEN, self.h_dim])

            for t in xrange(1, self.MAXLEN):

                h_t = self.H_transposed[t:t + 1, :, ]

                h_t_reshaped = tf.reshape(h_t, shape = [self.batch_size, self.h_dim])

                h_t_repeat = tf.expand_dims(h_t_reshaped, 1)

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

            self.h_star.append(tf.tanh(tf.add(self.Wpr, self.Wxhn), name = "hstar"))

            self.Y = self.y_output  # its length must be x_length

            self.H = self.x_output  # the set of hidden states from Q2

        self.hstar_two_way = tf.concat(self.h_star, axis = 1)

        self.h_star_x_cnn = get_cnn_embedding(self.x_emb, self.dropout_keep_prob, self.MAXLEN, self.dim, self.filter_sizes, self.num_filters)

        print('h_star_x_cnn: ', self.h_star_x_cnn.shape)

        self.h_star_y_cnn = get_cnn_embedding(self.y_emb, self.dropout_keep_prob, self.MAXLEN, self.dim, self.filter_sizes, self.num_filters)

        print('h_star_y_cnn: ', self.h_star_y_cnn.shape)

        self.h_layer1 = self.dense_batch_prelu(self.hstar_two_way, 600, self.dropout_keep_prob, self.is_training, "hidden1")

        self.h_layer2 = self.dense_batch_prelu(self.h_layer1, 300, self.dropout_keep_prob, self.is_training, "hidden2")

	self.h_layer3 = self.dense_batch_prelu(self.h_layer2, 150, self.dropout_keep_prob, self.is_training, "hidden3")

        self.W_pred = tf.get_variable("W_pred", shape = [ 150, 2 ])

        self.scaled_pred = tf.nn.softmax(tf.matmul(self.h_layer3, self.W_pred), name = "pred_layer")

        self.unscaled_pred = tf.matmul(self.h_layer3, self.W_pred)
        # print "pred",self.pred,"target",self.target

        class_weights = tf.div(tf.reduce_sum(self.target, 0), tf.constant(float(self.batch_size)))

        weighted_logits = tf.multiply(self.unscaled_pred, class_weights)  # shape [batch_size, 2]

        self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = weighted_logits, labels = self.target, name = "loss"))

        self.predictions = tf.argmax(self.scaled_pred, 1)

        self.predictions_probs = self.scaled_pred[:, 1]

        correct = tf.equal(self.predictions, tf.argmax(self.target, 1))

        self.acc = tf.reduce_mean(tf.cast(correct, "float"), name = "accuracy")

        self.optimizer = tf.train.AdamOptimizer()

        self.optim = self.optimizer.minimize(self.loss, var_list = tf.trainable_variables())

        _ = tf.summary.scalar("loss", self.loss)

    def dense_batch_prelu(self, x, number_of_hidden_units, dropout_prob, phase, scope):
    	with tf.variable_scope(scope):
                print('dropout', dropout_prob)

        	h1 = tf.contrib.layers.fully_connected(x, number_of_hidden_units,
                                               activation_fn = None,
                                               scope = 'dense')
        	h2 = tf.contrib.layers.batch_norm(h1,
                                          center = True, scale = True,
                                          is_training = phase,
                                          scope = 'bn')
        	h3 = tf.nn.relu(h2, 'relu')

 	        output = tf.nn.dropout(h3, dropout_prob)

		return output


    def train(self, \
              xdata, ydata, zdata, x_lengths, y_lengths, \
              xdevdata, ydevdata, zdevdata, xdev_lengths, ydev_lengths, \
              xxdata, yydata, zzdata, xx_lengths, yy_lengths, \
              MAXITER):

        merged_sum = tf.summary.merge_all()

        # writer = tf.train.SummaryWriter("./logs/%s" % "modeldir", self.sess.graph_def)

        tf.initialize_all_variables().run()

        start_time = time.time()

        best_val_loss = 1e100
	best_val_acc = 0.0

        for ITER in range(MAXITER):
	    total_acc = 0.0
            print('**************EPOCH****************\n', str(ITER))
            epoch_start_time = time.time()
            total_loss = 0
            # xdata, ydata, zdata, x_lengths, y_lengths = joint_shuffle(xdata, ydata, zdata, x_lengths, y_lengths)
            for i in xrange(0, len(xdata), self.batch_size):
                x, y, z, xlen, ylen = xdata[i:i + self.batch_size], \
                                ydata[i:i + self.batch_size], \
                                zdata[i:i + self.batch_size], \
                                x_lengths[i:i + self.batch_size], \
                                y_lengths[i:i + self.batch_size]

                x, y, z, xlen, ylen = augment_data(x, y, z, xlen, ylen)

                feed_dict = {self.x: x, \
                             self.y: y, \
                             self.target: z, \
                             self.x_length:xlen, \
                             self.y_length:ylen, \
 		             self.is_training:1, \
			     self.dropout_keep_prob:0.5 }

                att, _ , loss, acc, summ = self.sess.run([self.att, self.optim, self.loss, self.acc, merged_sum], feed_dict = feed_dict)

                total_loss += loss
		total_acc += acc

            print("Epoch Time: ", time.time() - epoch_start_time)

            total_loss = total_loss / float(len(xdata))
	    total_acc = total_acc / float(len(xdata) / self.batch_size)
            print ("Loss", total_loss, "Accuracy On Training", total_acc)

            total_val_loss, total_val_acc = self.validate(xdevdata, ydevdata, zdevdata, xdev_lengths, ydev_lengths, ITER)

            if(best_val_loss >= total_val_loss or best_val_acc <= total_val_acc):
		if (best_val_loss >= total_val_loss):
			best_val_loss = total_val_loss
		if (best_val_acc <= total_val_acc):
			best_val_acc = total_val_acc
                self.test(xxdata, yydata, zzdata, xx_lengths, yy_lengths, ITER)

        elapsed_time = time.time() - start_time

        print("Total Time", elapsed_time)

    def test(self, \
              xxdata, yydata, zzdata, xx_lengths, yy_lengths, epoch_number):

        merged_sum = tf.summary.merge_all()

        start_time = time.time()

        print('**********TESTING STARTED**********')
        test_predictions = [0] * len(xxdata)
        total_test_loss = 0
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
			  self.is_training:0, \
                          self.dropout_keep_prob:1}

            test_loss, att, test_acc, summ, test_predictions[i : i + self.batch_size] = self.sess.run([self.loss, self.att, self.acc, merged_sum, self.predictions_probs], feed_dict = tfeed_dict)
            total_test_loss += test_loss
	total_test_loss /= float(len(xxdata))
            # print ('Test batches processed: ', (i / batch_size))

            # test_predictions.extend(test_preds)
        print('............Test loss.....', total_test_loss)
        print('**********TESTING ENDED**********')

        write_submission_file(test_predictions, '../data/submissions', epoch_number)

        elapsed_time = time.time() - start_time

        print("Total Time", elapsed_time)

    def plot_calibration_graph(self, zzdata, predictions, ITER, step = 0.1):

        bins = []
        i = 0
        while i <= 1.0:
            bins.append(i)
            i += step

        bin_indices = np.digitize(predictions, bins)
        bin_counts = np.zeros(len(bins))
        bin_duplicate_counts = np.zeros(len(bins))
        bin_duplicate_probs = np.zeros(len(bins))

        for idx, prediction in enumerate(predictions):
            bin_idx = bin_indices[idx]
            bin_counts[bin_idx] += 1
            if zzdata[idx][1] == 1:
                bin_duplicate_counts[bin_idx] += 1

        eps = 1e-8
        for bin_idx, each_bin in enumerate(bins):
            bin_duplicate_probs[bin_idx] = bin_duplicate_counts[bin_idx] / float(bin_counts[bin_idx] + eps)

        plt.plot(bins, bin_duplicate_probs)
        plt.savefig('calibration' + str(ITER) + '.png')
        plt.clf()


    def validate(self, \
              xxdata, yydata, zzdata, xx_lengths, yy_lengths, ITER):

        merged_sum = tf.summary.merge_all()

        start_time = time.time()

        print('**********VALIDATION STARTED**********')
        test_predictions = [0] * len(xxdata)

	total_val_loss = 0
	total_val_acc = 0

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
			  self.is_training:0, \
			  self.dropout_keep_prob:1}

            test_loss, att, test_acc, summ, test_predictions[i : i + self.batch_size] = self.sess.run([self.loss, self.att, self.acc, merged_sum, self.predictions_probs], feed_dict = tfeed_dict)

	    total_val_loss += test_loss
	    total_val_acc += test_acc

        total_val_loss /= float(len(xxdata))
	total_val_acc /= float(len(xxdata) / self.batch_size)
	

	# self.plot_calibration_graph(zzdata, test_predictions, ITER, 0.1)

	print('*****Validation Accuracy********', total_val_acc)

        print('...........Validation loss.....', total_val_loss)
        print('**********VALIDATION ENDED**********')



        elapsed_time = time.time() - start_time

        print("Total Time", elapsed_time)

        return total_val_loss, total_val_acc

def joint_shuffle(xdata, ydata, zdata, x_lengths, y_lengths):

    tmp = list(zip(xdata, ydata, zdata, x_lengths, y_lengths))

    random.shuffle(tmp)

    xdata, ydata, zdata, x_lengths, y_lengths = zip(*tmp)

    return xdata, ydata, zdata, x_lengths, y_lengths

if __name__ == "__main__":

    options = get_params()

    X_train, Y_train, Z_train, X_dev, Y_dev, Z_dev, vocab_dict, glove_matrix = fitData()

    test_ids, X_test, Y_test, Z_test = fit_test_data()

    X_train_lengths = [len(x) for x in X_train]

    X_train_lengths = np.asarray([len(x) for x in X_train]).reshape(len(X_train))

    X_dev_lengths = [len(x) for x in X_dev]

    X_dev_lengths = np.asarray([len(x) for x in X_dev]).reshape(len(X_dev))

    X_test_lengths = [len(x) for x in X_test]

    X_test_lengths = np.asarray([len(x) for x in X_test]).reshape(len(X_test))



    Y_train_lengths = [len(x) for x in Y_train]

    Y_train_lengths = np.asarray([len(x) for x in Y_train]).reshape(len(Y_train))

    Y_dev_lengths = [len(x) for x in Y_dev]

    Y_dev_lengths = np.asarray([len(x) for x in Y_dev]).reshape(len(Y_dev))

    Y_test_lengths = [len(x) for x in Y_test]

    Y_test_lengths = np.asarray([len(x) for x in Y_test]).reshape(len(Y_test))



    Z_train = np_utils.to_categorical(Z_train, nb_classes = options.num_classes)

    Z_test = np_utils.to_categorical(Z_test, nb_classes = options.num_classes)

    Z_dev = np_utils.to_categorical(Z_dev, nb_classes = options.num_classes)

    MAXLEN = options.maxlen

    MAXITER = options.epochs

    with tf.Session() as sess:

        model = AttentionModel(options, sess, MAXLEN, vocab_dict, glove_matrix, batch_size = 512)

        model.build_model()

        model.train(X_train, Y_train, Z_train, X_train_lengths, Y_train_lengths, \
                    X_dev, Y_dev, Z_dev, X_dev_lengths, Y_dev_lengths, \
                    X_test, Y_test, Z_test, X_test_lengths, Y_test_lengths, \
                    MAXITER)

#         model.test(X_test, Y_test, Z_test, X_test_lengths, Y_test_lengths, \
#               glove_matrix)
