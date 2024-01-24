from sentimental_hwglu.sa_model import SentimentAnalysisPipeline
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

class KerasTokenizer(SentimentAnalysisPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._max_features = 500
        self._max_length = None
        self._tokeniezer = None
        self.set_params(**kwargs)
    
    def get_params(self, deep=None):
        return {
            "max_features": self._max_features,
            "max_length": self._max_length,
        }
    
    def set_params(self, **kwargs):
        if 'max_features' in kwargs:
            self._max_features = kwargs.get("max_features")
        if 'max_length' in kwargs:
            self._max_length = kwargs.get('max_length')
        return self
    
    def fit(self, X, y, **fitparmas):
        self._tokenizer = Tokenizer(num_words=self._max_features)
        self._tokenizer.fit_on_texts(X)
        return self
    
    def transform(self, X):
        sequences = self._tokenizer.texts_to_sequences(X)
        reviews = pad_sequences(sequences, maxlen=self._max_length)
        return reviews


class SA_LSTM(SentimentAnalysisPipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self._max_words = 500
        self._max_len = None
        self._epoches = 5
        self._verbose = 0
        self.set_params(**params)
        self._model = None
        self._y_pred = []

    def _tokenizeData(self, data, max_words=500, max_len=None):
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        reviews = pad_sequences(sequences, maxlen=max_len)
        print(reviews.shape)
        print(reviews)
        return reviews

    def fit(self, X, y, **fitparmas):
        if self._model is None:
            self._model = Sequential()
            self._model.add(layers.Embedding(self._max_words, 20)) #The embedding layer
            self._model.add(layers.Bidirectional(layers.LSTM(64, name='lstm'), name='lstm-bidir'))
            self._model.add(layers.Dense(64,activation='relu'))
            self._model.add(layers.Dense(2,activation='softmax'))
            self._model.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
        y = tf.keras.utils.to_categorical(y, 2, dtype="float32")
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.25, random_state=self._random_state)
        if self._verbose > 3: print("train", X_train.shape, " -> ", X_train, ", max: ", np.max(X_train))
        if self._verbose > 3: print("validate", X_validate.shape, " -> ", X_validate, ", max: ", np.max(X_validate))
        self._history = self._model.fit(X_train, y_train, epochs=self._epoches, validation_data=(X_validate, y_validate))

    def predict(self, X, **predict_params):
        if self._verbose > 3: print("[PREDICT] ================================ ")
        if self._verbose > 3: print("[PREDICT] try on ", X.shape, ", params: ", self.get_params())
        self._y_pred = self._model.predict(X)
        if self._verbose > 3: print("[PREDICT]", self._y_pred, -1)
        if self._verbose > 3: print("[PREDICT]", np.argmax(self._y_pred, -1))
        return np.argmax(self._y_pred, -1)

    def set_params(self, **kwargs):
        changed = False
        if 'max_words' in kwargs:
            v = kwargs.get('max_words', 500)
            if self._max_words != v:
                changed = True
            self._max_words = v
        if 'max_len' in kwargs:
            v = kwargs.get('max_len', None)
            if self._max_len != v:
                changed = True
            self._max_len = v
        if 'epoches' in kwargs:
            v = kwargs.get('epoches', 5)
            if self._epoches != v:
                changed = True
            self._epoches = v
        if changed:
            self._model = None
        try: self._verbose = kwargs.get('verbose', 0)
        except: pass
        if self._verbose > 3: print("[SET PARAMS] ", kwargs)
        return self
    
    def get_params(self, deep: bool = True) -> dict:
        return {
            "max_words": self._max_words,
            "max_length": self._max_len,
            "epoches": self._epoches,
            "verbose": self._verbose
        }

    def __str__(self) -> str:
        return "LSTM"

    def summary(self)-> str:
        self._model.summary()

class LSTMPipeline(SentimentAnalysisPipeline):
    def __init__(self, **params):
        super().__init__(**params)
        self._params = params
        self._pipeline.steps.append(('vect', KerasTokenizer()))
        # self._pipeline.steps.append(('to_dense', DenseTransformer()))
        self._pipeline.steps.append(('lstm_sa', SA_LSTM()))
        self._pipeline.set_params(**params)
    
    def get_params(self, deep=None):
        return self._pipeline.get_params(deep)

    def set_params(self, **params):
        if 'vect__max_features' in params:
            features = params.get('vect__max_features')
            print("[LSTM_PIPELINE] setting lstm_sa_max_words to ", features)
            params['lstm_sa__max_words'] = features
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
        return "LSTMPipeline"

    def __str__(self) -> str:
        return "LSTMPipelineW2V"