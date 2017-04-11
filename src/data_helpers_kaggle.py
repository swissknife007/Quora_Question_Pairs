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

vocab_processor = None


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

def read_test_data(fileName):

    data = pd.read_csv(fileName, sep = ',')

    test_ids = list(data.test_id.values)

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

    return test_ids[:], questions1, questions2

def read_embeddings(word_index):
    embeddings_index = {}
    f = open('../data/glove.840B.300d.txt')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
        #if len(embeddings_index) > 1:
        #   break
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # +1 for the padding token
    embedding_matrix = np.zeros((len(word_index), 300))
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def generate_rsample(X, y, batch_size):
    # print len(X)
    X_psize = len(X) % batch_size
    # print X_psize
    rindices = random.sample(range(0, len(X)), batch_size - X_psize)
    # print len(rindices)
    y_dash = [y[index] for index in rindices]
    y_dash = np.array(y_dash)
    y = np.concatenate((y, y_dash))
    X = X + [X[index] for index in rindices]
    # print len(X)
    assert len(X) % batch_size == 0, 'len(X) is not a multiple of batch size!'
    return X, y

def fitData(fileName = '../data/train.csv', max_len = 40, batch_size = 512):

    questions1, questions2, y = readData(fileName)

    _, test_q1, test_q2 = read_test_data('../data/test.csv')

    global vocab_processor

    vocab_processor = VocabularyProcessor(max_len)

    vocab_processor.fit(questions1 + questions2 + test_q1 + test_q2)

    X_q1 = np.array(list(vocab_processor.transform(questions1)))

    X_q2 = np.array(list(vocab_processor.transform(questions2)))

    vocab_dict = vocab_processor.vocabulary_._mapping

    glove_matrix = read_embeddings(vocab_dict)

    print type(vocab_dict)

    all_data = zip(X_q1, X_q2)

    X_train, _, y_train, _ = train_test_split(all_data, y, test_size = 0, random_state = 42)

    X_train, y_train = generate_rsample(X_train, y_train, batch_size)

    X_train_q1, X_train_q2 = zip(*X_train)


    print 'len(X_train_q1): ', len(X_train_q1)

    print 'len(X_train_q2): ', len(X_train_q2)

    return X_train_q1, X_train_q2, y_train, vocab_dict, glove_matrix



def fit_test_data(fileName = '../data/test.csv', max_len = 40, batch_size = 512):

    test_ids, questions1, questions2 = read_test_data(fileName)

    global vocab_processor

    X_q1 = np.array(list(vocab_processor.transform(questions1)))

    X_q2 = np.array(list(vocab_processor.transform(questions2)))

    Y_test = np.zeros(X_q1.shape[0])

    X_test = zip(test_ids, X_q1, X_q2)

    X_test, Y_test = generate_rsample(X_test, Y_test, batch_size)

    test_id, X_test_q1, X_test_q2 = zip(*X_test)


    print 'len(X_test_q1): ', len(X_test_q1)
    print 'len(X_test_q2): ', len(X_test_q2)

    return test_ids, X_test_q1, X_test_q2, Y_test


def write_submission_file(predictions, fileName = '../data/submissions', epoch = ''):


    total_lines = min(2345795, len(predictions))
    output_file = open(fileName + "_" + str(epoch) + '.csv', 'w')

    output_file.write("test_id,is_duplicate\n")

    for test_id, prediction in enumerate(predictions):
        output_file.write(str(test_id) + "," + str(prediction) + "\n")
        if(test_id == total_lines):
            break
    output_file.close()
