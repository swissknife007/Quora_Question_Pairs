import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import Lambda
from keras.utils import np_utils
from keras.engine.topology import Merge
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text

def read_train_data():
    data = pd.read_csv('../../data/train.csv', sep = ',')
    # "id","qid1","qid2","question1","question2","is_duplicate"
    # id = data.id.value
    # qid1 = data.qid1.value
    # qid2 = data.qid2.value
    question1 = data.question1.values
    question2 = data.question2.values
    y = data.is_duplicate.values

    # return id, qid1, qid2, question1, question2, y
    return question1, question2, y

def read_test_data():

    data = pd.read_csv('../../data/test.csv', sep = ',')
    # "test_id","question1","question2"
    test_id = data.test_id.values
    question1 = data.question1.values
    question2 = data.question2.values

    return test_id, question1, question2

def write_submission_file(predictions, fileName = '../data/submissions', epoch = ''):

    total_lines = min(2345795, len(predictions))
    output_file = open(fileName + "_" + str(epoch) + '.csv', 'w')

    output_file.write("test_id,is_duplicate\n")

    for test_id, prediction in enumerate(predictions):
        output_file.write(str(test_id) + "," + str(prediction) + "\n")
        if(test_id == total_lines):
            break
    output_file.close()

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis = 1, keepdims = True), K.epsilon()))

def dot_prod(vects):
    x, y = vects
    return K.sum(x * y, axis = 1, keepdims = True)

train_question1, train_question2, y = read_train_data()

test_id, test_question1, test_question2 = read_test_data()

tk = text.Tokenizer(nb_words = 300000)

max_len = 40

tk.fit_on_texts(list(train_question1.astype(str)) + list(train_question2.astype(str)) + list(test_question1.astype(str)) + list(test_question2.astype(str)))

train_x1 = tk.texts_to_sequences(train_question1.astype(str))

train_x1 = sequence.pad_sequences(train_x1, maxlen = max_len)

train_x2 = tk.texts_to_sequences(train_question2.astype(str))

train_x2 = sequence.pad_sequences(train_x2, maxlen = max_len)

test_x1 = tk.texts_to_sequences(test_question1.astype(str))

test_x1 = sequence.pad_sequences(test_x1, maxlen = max_len)

test_x2 = tk.texts_to_sequences(test_question2.astype(str))

test_x2 = sequence.pad_sequences(test_x2, maxlen = max_len)

word_index = tk.word_index

ytrain_enc = np_utils.to_categorical(y)

embeddings_index = {}
f = open('../../data/glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
    #if len(embeddings_index) > 10:
    #    break
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_features = 200000
filter_length = 5
nb_filter = 64
pool_length = 4

model = Sequential()
print('Build model...')

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1, 300, input_length = 40, dropout = 0.2))
model1.add(LSTM(300, dropout_W = 0.2, dropout_U = 0.2))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1, 300, input_length = 40, dropout = 0.2))
model2.add(LSTM(300, dropout_W = 0.2, dropout_U = 0.2))

'''
distance_model = Sequential()
distance_model.add(Lambda(euclidean_distance)([model1, model2]))


angle_model = Sequential()
angle_model.add(Lambda(dot_prod)([model1, model2]))
'''
merged_model = Sequential()

merged_model.add(Merge([model1, model2], mode = 'concat'))

merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(300))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())

merged_model.add(Dense(1))
merged_model.add(Activation('sigmoid'))

merged_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

checkpoint = ModelCheckpoint('weights.h5', monitor = 'val_acc', save_best_only = True, verbose = 2)

merged_model.fit([train_x1, train_x2], y = y, batch_size = 512, nb_epoch = 5,
                 verbose = 1, validation_split = 0.1, shuffle = True, callbacks = [checkpoint])

proba_preds = merged_model.predict_proba([test_x1, test_x2], batch_size = 512)

proba_preds = proba_preds[:]

write_submission_file(proba_preds, fileName = '../../data/keras_submissions')
