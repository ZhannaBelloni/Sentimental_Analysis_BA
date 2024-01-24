from sentimental_hwglu.sa_model import SentimentAnalysisPipeline
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
## Check me!!!

class SA_RNN_Pipeline(SentimentAnalysisPipeline):
    def __init__(self, random_state=None, max_words=500, max_length=None, epochs=5):
        super().__init__(random_state)
        self._max_words = max_words
        self._max_len = max_length
        self._epoches = epochs
        self._pipeline = Sequential()
        self._pipeline.add(layers.Embedding(max_words, 20)) #The embedding layer
        self._pipeline.add(layers.RNN(15,dropout=0.5)) #Our LSTM layer
        self._pipeline.add(layers.Dense(2,activation='softmax'))
        self._pipeline.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
        self._y_pred = []

    def _tokenizeData(self, data, max_words=500, max_len=None):
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(data)
        sequences = tokenizer.texts_to_sequences(data)
        reviews = pad_sequences(sequences, maxlen=max_len)
        return reviews

    def fit(self, X, y, **fitparmas):
        X = self._tokenizeData(X, self._max_words, max_len=self._max_len)
        y = tf.keras.utils.to_categorical(y, 2, dtype="float32")
        # if validation_data:
        #    X_test = self._tokenizeData(validation_data[0])
        #    y_test = tf.keras.utils.to_categorical(validation_data[1], 2, dtype="float32")
        #    self._pipeline.fit(X, y, epochs=self._epoches, validation_data=(X_test, y_test))
        #else:
        X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.25, random_state=self._random_state)
        self._pipeline.fit(X, y, epochs=self._epoches, validation_data=(X_validate, y_validate))

    def predict(self, X, **predict_params):
        X = self._tokenizeData(X, self._max_words, max_len=self._max_len)
        self._y_pred = self._pipeline.predict(X)
        return np.argmax(self._y_pred, -1)

    def set_params(self, **kwargs):
        self._params = kwargs

    def __str__(self) -> str:
        return "RNN"

    def summary(self)-> str:
        self._pipeline.summary()