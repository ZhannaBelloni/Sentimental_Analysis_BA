import threading
from sentimental_hwglu.sa_model import SentimentAnalysisPipeline
from sentimental_hwglu.utils import loadGoogleW2v
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator
from sentimental_hwglu.sa_embeddings import Word2VectSA
from sklearn.pipeline import Pipeline

from tensorflow import keras
import tensorflow as tf

import numpy as np

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.models import Sequential
from keras import layers
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from mlxtend.preprocessing import DenseTransformer

from sentimental_hwglu.globals import w2v, lock_w2v


class SA_LSTM_W2V(SentimentAnalysisPipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self._max_words = 500
        self._max_len = None
        self._epoches = 5
        self._verbose = 0
        self._model = None
        self._tokenizer = None
        self._y_pred = []
        self._embedded_dim = 300
        self._w2v = None
        self._model_path = None
        self._max_length = None
        self._words_path = None
        self._bidirectional = False
        self._layer_1 = True
        self._layer_1_size = True
        self._layer_1_func = "relu"
        self._output_func = "relu"
        self.set_params(**params)
    
    def __prepare_matrix_embedding(self, vocab_size):
        embed_matrix=np.zeros(shape=(vocab_size, self._embedded_dim))
        word_vec_dict={}
        for word in self._w2v.key_to_index.keys():
            word_vec_dict[word]=self._w2v[word]
        n = 0
        tot = 0
        for word,i in self._tokenizer.word_index.items():
            tot += 1
            embed_vector=word_vec_dict.get(word)
            if embed_vector is not None:  # word is in the vocabulary learned by the w2v model
                embed_matrix[i]=embed_vector
                n += 1
        print("[INFO _prepare matrix: ", n, "/", tot)
        return embed_matrix

    # TODO: see raschka2021 - p. 626
    # "lstm_sa__rnn_type": ["LSTM", "RNN"],
    def fit(self, X, y, **fitparmas):
        global w2v, lock_w2v
        with lock_w2v:
            if self._model_path is not None and self._model_path not in w2v:
                if self._verbose > 0: print(" loading model from '{}'".format(self._model_path))
                w2v[self._model_path] = loadGoogleW2v(self._model_path)
        self._w2v = w2v[self._model_path]
        if self._tokenizer is None:
            self._tokenizer = Tokenizer(num_words=self._max_words)
            self._tokenizer.fit_on_texts(X)
            all_words = []
            with open(self._words_path, 'r') as f:
                for l in f:
                    all_words.append(l.strip())
            self._tokenizer.fit_on_texts(all_words)
        if self._model is None:
            vocab_size = len(self._tokenizer.word_index) + 1
            embed_matrix = self.__prepare_matrix_embedding(vocab_size)
            self._model = Sequential()
            self._model.add(layers.Embedding(input_dim=vocab_size,output_dim=self._embedded_dim, embeddings_initializer=keras.initializers.Constant(embed_matrix)))
            if self._bidirectional:
                self._model.add(layers.Bidirectional(layers.LSTM(64, name='lstm'), name='lstm-bidir'))
            else:
                self._model.add(layers.LSTM(64, name='lstm'))
            if self._layer_1:
                self._model.add(layers.Dense(self._layer_1_size,activation=self._layer_1_func))
            self._model.add(layers.Dense(2,activation=self._output_func))
            self._model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])

        X = self._transform_text(X)
        y = tf.keras.utils.to_categorical(y, 2, dtype="float32")
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.25, random_state=self._random_state)
        if self._verbose > 3: print("train", X_train.shape)
        if self._verbose > 3: print("validate", X_validate.shape)
        self._history = self._model.fit(X_train, y_train, epochs=self._epoches, validation_data=(X_validate, y_validate))

    def _transform_text(self, X):
        sequences = self._tokenizer.texts_to_sequences(X)
        reviews = pad_sequences(sequences, maxlen=self._max_length)
        return reviews

    def predict(self, X, **predict_params):
        if self._verbose > 3: print("[PREDICT] ================================ ")
        if self._verbose > 3: print("[PREDICT] try on ", X.shape, ", params: ", self.get_params())
        X = self._transform_text(X)
        self._y_pred = self._model.predict(X)
        if self._verbose > 3: print("[PREDICT]", self._y_pred, -1)
        if self._verbose > 3: print("[PREDICT]", np.argmax(self._y_pred, -1))
        return np.argmax(self._y_pred, -1)

    def set_params(self, **kwargs):
        changed = False
        if 'layer_1' in kwargs:
            v = kwargs.get('layer_1')
            if self._layer_1 != v: changed = True
            self._layer_1 = v
        if 'layer_1_size' in kwargs:
            v = kwargs.get('layer_1_size')
            if self._layer_1_size != v: changed = True
            self._layer_1_size = v
        if 'layer_1_activation' in kwargs:
            v = kwargs.get('layer_1_activation')
            if self._layer_1_func != v: changed = True
            self._layer_1_func = v
        if 'output_activation' in kwargs:
            v = kwargs.get('output_activation')
            if self._output_func != v: changed = True
            self._output_func = v
        if 'bidirectional' in kwargs:
            v = kwargs.get('bidirectional')
            if self._bidirectional != v: changed = True
            self._bidirectional = v
        if 'model' in kwargs:
            v = kwargs.get('model')
            if self._model_path != v: changed = True
            self._model_path = v
        if 'words' in kwargs:
            v = kwargs.get('words', None)
            if self._words_path != v: changed = True
            self._words_path = v
        if 'max_words' in kwargs:
            v = kwargs.get('max_words', 500)
            if self._max_words != v: changed = True
            self._max_words = v
        if 'max_len' in kwargs:
            v = kwargs.get('max_len', None)
            if self._max_len != v: changed = True
            self._max_len = v
        if 'epoches' in kwargs:
            v = kwargs.get('epoches', 5)
            if self._epoches != v: changed = True
            self._epoches = v
        if changed:
            self._model = None
            self._tokenizer = None
        try: self._verbose = kwargs.get('verbose', 0)
        except: pass
        if self._verbose > 3: print("[SET PARAMS] ", kwargs)
        return self
    
    def get_params(self, deep: bool = True) -> dict:
        return {
            "max_words": self._max_words,
            "max_length": self._max_len,
            "epoches": self._epoches,
            "verbose": self._verbose,
            "model": self._model_path,
            "words": self._words_path,
            "bidirectional": self._bidirectional,
            "layer_1": self._layer_1,
            "layer_1_size": self._layer_1_size,
            "layer_1_activation": self._layer_1_func,
            "output_activation": self._output_func
        }

    def __str__(self) -> str:
        return "LSTM"

    def summary(self)-> str:
        self._model.summary()

class LSTMPipeline_W2V(SentimentAnalysisPipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self._params = params
        self._pipeline.steps.append(('lstm_sa', SA_LSTM_W2V()))
        self._pipeline.set_params(**params)
    
    def get_params(self, deep=None):
        return self._pipeline.get_params(deep)

    def set_params(self, **params):
        self._pipeline.set_params(**params)
        return self

    def save(self, fileout):
        if fileout is None: return
        fileout_name = '{}_{}.json'.format(fileout, str(self))
        with open(fileout_name, 'a') as f: f.write("[")
        first = True
        for p in self._pipeline:
            if not first:
                with open(fileout_name, 'a') as f: f.write(",")
            else: first = False
            p.save(fileout_name)
        with open(fileout_name, 'a') as f: f.write("]")

    def __str__(self) -> str:
        return "LSTMPipeline_W2V"
