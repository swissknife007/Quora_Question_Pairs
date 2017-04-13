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
    return question1[:], question2[:], y[:]

def read_test_data():

    data = pd.read_csv('../../data/test.csv', sep = ',')
    # "test_id","question1","question2"
    test_id = data.test_id.values
    question1 = data.question1.values
    question2 = data.question2.values

    return test_id[:], question1[:], question2[:]

def write_submission_file(predictions, fileName = '../data/submissions', epoch = ''):


    total_lines = min(2345795, len(predictions))
    output_file = open(fileName + "_" + str(epoch) + '.csv', 'w')

    output_file.write("test_id,is_duplicate\n")

    for test_id, prediction in enumerate(predictions):
	print prediction
        output_file.write(str(test_id) + "," + str( 1.0 - prediction[0]) + "\n")
        if(test_id == total_lines):
            break
    output_file.close()

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
    if len(embeddings_index) > 10:
        break
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
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights = [embedding_matrix],
                     input_length = 40,
                     trainable = False))

model1.add(TimeDistributed(Dense(300, activation = 'relu')))
model1.add(Lambda(lambda x: K.sum(x, axis = 1), output_shape = (300,)))

model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights = [embedding_matrix],
                     input_length = 40,
                     trainable = False))

model2.add(TimeDistributed(Dense(300, activation = 'relu')))
model2.add(Lambda(lambda x: K.sum(x, axis = 1), output_shape = (300,)))

model3 = Sequential()
model3.add(Embedding(len(word_index) + 1,
                     300,
                     weights = [embedding_matrix],
                     input_length = 40,
                     trainable = False))
model3.add(Convolution1D(nb_filter = nb_filter,
                         filter_length = filter_length,
                         border_mode = 'valid',
                         activation = 'relu',
                         subsample_length = 1))
model3.add(Dropout(0.2))

model3.add(Convolution1D(nb_filter = nb_filter,
                         filter_length = filter_length,
                         border_mode = 'valid',
                         activation = 'relu',
                         subsample_length = 1))

model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))

model3.add(Dense(300))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())

model4 = Sequential()
model4.add(Embedding(len(word_index) + 1,
                     300,
                     weights = [embedding_matrix],
                     input_length = 40,
                     trainable = False))
model4.add(Convolution1D(nb_filter = nb_filter,
                         filter_length = filter_length,
                         border_mode = 'valid',
                         activation = 'relu',
                         subsample_length = 1))
model4.add(Dropout(0.2))

model4.add(Convolution1D(nb_filter = nb_filter,
                         filter_length = filter_length,
                         border_mode = 'valid',
                         activation = 'relu',
                         subsample_length = 1))

model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.2))

model4.add(Dense(300))
model4.add(Dropout(0.2))
model4.add(BatchNormalization())
model5 = Sequential()
model5.add(Embedding(len(word_index) + 1, 300, input_length = 40, dropout = 0.2))
model5.add(LSTM(300, dropout_W = 0.2, dropout_U = 0.2))

model6 = Sequential()
model6.add(Embedding(len(word_index) + 1, 300, input_length = 40, dropout = 0.2))
model6.add(LSTM(300, dropout_W = 0.2, dropout_U = 0.2))

merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4, model5, model6], mode = 'concat'))
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

merged_model.load_weights("weights.h5")

proba_preds = merged_model.predict_proba([test_x1, test_x2, test_x1, test_x2, test_x1, test_x2], batch_size = 512)

write_submission_file(proba_preds, fileName = '../../data/keras_submissions')	
