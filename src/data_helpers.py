import numpy as np
import re
import itertools
from collections import Counter
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.preprocessing import text
from tensorflow.contrib.learn.python.learn.preprocessing import CategoricalVocabulary
from tensorflow.contrib.learn.python.learn.preprocessing import VocabularyProcessor
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random

def readData(fileName):
    data = pd.read_csv(fileName, sep = ',')
    y = data.is_duplicate.values[:]

    questions1 = list(data.question1.values)

    print type(questions1[0])

    print questions1[0]

    questions1 = [str(q)[:-1] + ' ?' for q in questions1[:]]

    print questions1[0]

    questions2 = list(data.question2.values)

    print type(questions2[0])

    print questions2[0]

    questions2 = [str(q)[:-1] + ' ?' for q in questions2[:]]

    print questions2[0]

    return questions1, questions2, y

def read_embeddings(word_index):
    embeddings_index = {}
    f = open('../data/glove.840B.300d.txt')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
        if len(embeddings_index) > 1000:
            break
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # +1 for the padding token
    embedding_matrix = np.zeros((len(word_index), 300))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def generate_rsample(X, batch_size):
    # print len(X)
    X_psize = len(X) % batch_size
    # print X_psize
    rindices = random.sample(range(0, len(X)), batch_size - X_psize)
    # print len(rindices)
    X = X + [X[index] for index in rindices]
    # print len(X)
    assert len(X) % batch_size == 0, 'len(X) is not a multiple of batch size!'
    return X

def fitData(fileName = '../data/kaggle_data/train.csv', max_len = 40, batch_size = 32):
    questions1, questions2, y = readData(fileName)
    vocab_processor = VocabularyProcessor(max_len)
    vocab_processor.fit(questions1 + questions2)
    X_q1 = np.array(list(vocab_processor.transform(questions1)))
    X_q2 = np.array(list(vocab_processor.transform(questions2)))
    # # Extract word:id mapping from the object.
    vocab_dict = vocab_processor.vocabulary_._mapping

    glove_matrix = read_embeddings(vocab_dict)

    print type(vocab_dict)

    all_data = zip(X_q1, X_q2)

    X_train, X_val, y_train, y_val = train_test_split(all_data, y, test_size = 0.30, random_state = 42)

    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.50, random_state = 42)

#     X_train = X_train[:100]
#     X_val = X_val[:100]
#     X_test = X_test[:100]

    X_train = generate_rsample(X_train, batch_size)
    X_val = generate_rsample(X_val, batch_size)
    X_test = generate_rsample(X_test, batch_size)

    X_train_q1, X_train_q2 = zip(*X_train)
    X_val_q1, X_val_q2 = zip(*X_val)
    X_test_q1, X_test_q2 = zip(*X_test)

    print 'len(X_train_q1): ', len(X_train_q1)
    print 'len(X_train_q2): ', len(X_train_q2)
    print 'len(X_test_q1): ', len(X_test_q1)
    print 'len(X_test_q2): ', len(X_test_q2)

    return X_train_q1, X_train_q2, X_val_q1, X_val_q2, X_test_q1, X_test_q2, y_train, y_val, y_test, vocab_dict, glove_matrix

# fitData()

# x_text = ['This is a cat', 'This must be boy', 'This is a a dog']
# max_document_length = max([len(x.split(" ")) for x in x_text])
#
# # # Create the vocabularyprocessor object, setting the max lengh of the documents.
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#
# # # Transform the documents using the vocabulary.
# x = np.array(list(vocab_processor.fit_transform(x_text)))
#
# # # Extract word:id mapping from the object.
# vocab_dict = vocab_processor.vocabulary_._mapping
#
# # # Sort the vocabulary dictionary on the basis of values(id).
# # # Both statements perform same task.
# # sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
# sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
#
# # # Treat the id's as index into list and create a list of words in the ascending order of id's
# # # word with id i goes at index i of the list.
# vocabulary = list(list(zip(*sorted_vocab))[0])
#
# print(vocabulary)
# print(x)
#
# def load_data_and_labels(positive_data_file, negative_data_file):
#     """
#     Loads MR polarity data from files, splits the data into words and generates labels.
#     Returns split sentences and labels.
#     """
#     # Load data from files
#     positive_examples = list(open(positive_data_file, "r").readlines())
#     positive_examples = [s.strip() for s in positive_examples]
#     negative_examples = list(open(negative_data_file, "r").readlines())
#     negative_examples = [s.strip() for s in negative_examples]
#     # Split by words
#     x_text = positive_examples + negative_examples
#     x_text = [clean_str(sent) for sent in x_text]
#     # Generate labels
#     positive_labels = [[0, 1] for _ in positive_examples]
#     negative_labels = [[1, 0] for _ in negative_examples]
#     y = np.concatenate([positive_labels, negative_labels], 0)
#     return [x_text, y]
#
#
# def batch_iter(data, batch_size, num_epochs, shuffle = True):
#     """
#     Generates a batch iterator for a dataset.
#     """
#     data = np.array(data)
#     data_size = len(data)
#     num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
#     for epoch in range(num_epochs):
#         # Shuffle the data at each epoch
#         if shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_data[start_index:end_index]

