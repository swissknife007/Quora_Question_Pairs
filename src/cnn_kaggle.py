import tensorflow as tf
import numpy as np

# x is the output of the embedding layer
def get_cnn_embedding(X, dropout_keep_prob, sequence_length, embedding_size, filter_sizes, num_filters):

        # dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        X_reshaped = tf.reshape(X, [-1, sequence_length, embedding_size, 1])

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name = "W")
                b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name = "b")
                conv = tf.nn.conv2d(
                    X_reshaped,
                    W,
                    strides = [1, 1, 1, 1],
                    padding = "VALID",
                    name = "conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize = [1, sequence_length - filter_size + 1, 1, 1],
                    strides = [1, 1, 1, 1],
                    padding = 'VALID',
                    name = "pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

        return h_drop