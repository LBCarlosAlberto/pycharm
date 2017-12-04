# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 11:28:34 2017

@author: Ning Zhang
"""

#word2vec

from gensim.models import word2vec
import logging
import nltk, string
from gensim import corpora
import os
import json
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Input, Flatten, Concatenate
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from Analyzer import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import classification_report
import pickle
def split_data(data, target, portion = 0.2, shuffle = True, random_state = 100):
        #split data into train and test by 0.8/0.2
    X_train, x_test, Y_train, y_test = train_test_split(\
                        data, target, test_size = 0.2, shuffle = shuffle,\
                        random_state = 0)
    return X_train, x_test, Y_train, y_test

def cnn_model(FILTER_SIZES, \
              # filter sizes as a list
              MAX_NB_WORDS, \
              # total number of words
              MAX_DOC_LEN, \
              # max words in a doc
              NUM_OUTPUT_UNITS=9, \
              # number of output units
              EMBEDDING_DIM=300, \
              # word vector dimension
              #32
              NUM_FILTERS=64, \
              # number of filters for all size
              DROP_OUT=0.5, \
              # dropout rate
              PRETRAINED_WORD_VECTOR=None,\
              # Whether to use pretrained word vectors
              LAM=0.01):            
              # regularization coefficient
    
    main_input = Input(shape=(MAX_DOC_LEN,), \
                       dtype='int32', name='main_input')
    
    if PRETRAINED_WORD_VECTOR is not None:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                        output_dim=EMBEDDING_DIM, \
                        input_length=MAX_DOC_LEN, \
                        weights=[PRETRAINED_WORD_VECTOR],\
                        trainable=False,\
                        name='embedding')(main_input)
    else:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                        output_dim=EMBEDDING_DIM, \
                        input_length=MAX_DOC_LEN, \
                        name='embedding')(main_input)
    # add convolution-pooling-flat block
    conv_blocks = []
    total_num_filters = 0
    for f in FILTER_SIZES:
        conv = Conv1D(filters=NUM_FILTERS, kernel_size=f, \
                      activation='relu', name='conv_'+str(f))(embed_1)
        conv = MaxPooling1D(MAX_DOC_LEN-f+1, name='max_'+str(f))(conv)
        conv = Flatten(name='flat_'+str(f))(conv)
        conv_blocks.append(conv)
        total_num_filters += NUM_FILTERS

    z=Concatenate(name='concate')(conv_blocks)
    drop=Dropout(rate=DROP_OUT, name='dropout')(z)

    dense = Dense(total_num_filters, activation='relu',\
                    kernel_regularizer=l2(LAM),name='dense')(drop)
    preds = Dense(NUM_OUTPUT_UNITS, activation='sigmoid', name='output')(dense)
    model = Model(inputs=main_input, outputs=preds)
    model.compile(loss="binary_crossentropy", \
              optimizer="adam", metrics=["accuracy"]) 
    return model

if __name__ == '__main__':
    topics = ['business','environment','fashion','lifeandstyle',\
                'politics','sport','technology','travel','world']
    print("load in data...")
    use_trained_model = False
    FILTER_SIZES = [2, 3, 4, 5, 6, 7, 8]
    MAX_NB_WORDS= 20000 #20000
    MAX_DOC_LEN= 2000 #1000 
    BEST_MODEL_FILEPATH = 'cnn_keras_pretrained_wv.hd5'
    BATCH_SIZES = 64
    NUM_EPOCHES = 10
    EMBEDDING_DIM = 300
    print("split data")
    if (os.path.isfile('x_train.pickle')):
        with open('x_train.pickle', 'rb') as f:
            x_train = pickle.load(f)
        with open('x_test.pickle', 'rb') as f:
            x_test = pickle.load(f)
        with open('y_train.pickle', 'rb') as f:
            y_train = pickle.load(f)
        with open('y_test.pickle', 'rb') as f:
            y_test = pickle.load(f)
    else:
        directory = "articles"
        i = 0
        articles = []
        tags = []
        topics_dict = {}
        for filename in os.listdir(directory):
            topics_dict[i] = filename.split('.')[0]
            if filename.endswith(".json"):
                with open(join(directory, filename), 'r') as f:
                    json_data = f.read()
                    data = json.loads(json_data)
                    for key, value in data.items():
                        value = value.lower()
                        #value = value.encode('ascii')
                        articles.append(value)
                        tags.append(i)
                i += 1
        #split the data
        x_train, x_test, y_train, y_test = split_data(articles, tags, portion = 0.2, shuffle = True, random_state = 100)
        #train word2vec model
        print("training wv model...")
        sentences_train = [[token.strip(string.punctuation).strip() \
             for token in nltk.word_tokenize(doc) \
                 if token not in string.punctuation and \
                 len(token.strip(string.punctuation).strip())>=2]\
             for doc in x_train]
    
        sentences_test = [[token.strip(string.punctuation).strip() \
             for token in nltk.word_tokenize(doc) \
                 if token not in string.punctuation and \
                 len(token.strip(string.punctuation).strip())>=2]\
             for doc in x_test]
        #sentences_train = [x.split(' ') for x in x_train]
        logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', \
                    level = logging.INFO)
        
        wv_model = word2vec.Word2Vec(sentences_train, min_count = 5, \
                             size = EMBEDDING_DIM, window = 5, workers = 6)
        wv_model.save('pretrained_w2v_model.dat')
        print("tokenizing...")
        #tokenize dataset
        x_train = [' '.join(sentence) for sentence in sentences_train]
        x_test = [' '.join(sentence) for sentence in sentences_test]
        with open('x_train.pickle', 'rb') as f:
            pickle.dump(x_train, f)
        with open('x_test.pickle', 'rb') as f:
            pickle.dump(x_test, f)
        with open('y_train.pickle', 'rb') as f:
            pickle.dump(y_train, f)
        with open('y_test.pickle', 'rb') as f:
            pickle.dump(y_test, f)
    print("check if pretrain tokenizer exists...")
    if os.path.isfile('pretrain_wv_tokenizer.pickle'):
        with open('pretrain_wv_tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        tokenizer = Tokenizer(MAX_NB_WORDS)
        tokenizer.fit_on_texts(x_train)
        with open('pretrain_wv_tokenizer.pickle', 'wb') as f:
            pickle.dump(tokenizer, f)
    
    test_sequences = tokenizer.texts_to_sequences(x_test)
    test_padded_sequences = pad_sequences(test_sequences, maxlen = MAX_DOC_LEN, padding = 'post', truncating = 'post')
    if not use_trained_model:
        print("training model...")
        voc = tokenizer.word_index
        train_sequences = tokenizer.texts_to_sequences(x_train)
        train_padded_sequences = pad_sequences(train_sequences, maxlen = MAX_DOC_LEN, padding = 'post', truncating = 'post')

        #convert labels to one hot
        one_hot_y_train = np_utils.to_categorical(y_train)
        # tokenizer.word_index provides the mapping 
        # between a word and word index for all words
        NUM_WORDS = min(MAX_NB_WORDS, len(voc))
        #process data for training word2vec
        if os.path.isfile('pretrained_w2v_model.dat'):
            wv_model = word2vec.Word2Vec.load('pretrained_w2v_model.dat')
    
        embedding_matrix = np.zeros((NUM_WORDS + 1, EMBEDDING_DIM))
        not_match_words_count = 0
        for word, i in tokenizer.word_index.items():
            if i >= NUM_WORDS:
                continue
            if word in wv_model.wv:
                embedding_matrix[i] = wv_model.wv[word]
            else:
                not_match_words_count += 1
        print("not_match_words_count:", not_match_words_count)
        NUM_OUTPUT_UNITS = len(np.unique(y_train))
        model = cnn_model(FILTER_SIZES, MAX_NB_WORDS, \
                  MAX_DOC_LEN, NUM_OUTPUT_UNITS, \
                  PRETRAINED_WORD_VECTOR = embedding_matrix)

        early_stop = EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
        checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_acc', \
                             verbose=2, save_best_only=True, mode='max')
    
        training=model.fit(train_padded_sequences, one_hot_y_train, \
            batch_size=BATCH_SIZES, epochs=NUM_EPOCHES, \
            callbacks=[early_stop, checkpoint],\
            validation_split=0.2, verbose=2)
    else:
        model = load_model('cnn_keras_pretrained_wv.hd5')

    #model.save('cnn_keras_pretrained_wv.hd5')
    predictions = model.predict(test_padded_sequences)
    predictions = predictions.argmax(axis = -1)
    print(classification_report(y_test, predictions, target_names=topics))
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, predictions)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(8,8))
    plot_confusion_matrix(cnf_matrix, classes=topics,
                  title='Confusion matrix, without normalization')