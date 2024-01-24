
import json
import threading
import numpy as np
from sentimental_hwglu.utils import loadGoogleW2v
from sklearn.base import BaseEstimator

from sentimental_hwglu.globals import w2v, lock_w2v

class Word2VectSA(BaseEstimator):
    def __init__(self, **params):
        global w2v, lock_w2v
        self.set_params(**params)
        with lock_w2v:
            if self._model_path is not None and self._model_path not in w2v:
                if self._verbose > 0: print(" loading model from '{}'".format(self._model_path))
                w2v[self._model_path] = loadGoogleW2v(self._model_path)

    def transform(self, X): 
        global w2v, lock_w2v
        with lock_w2v:
            if self._model_path is not None and self._model_path not in w2v:
                if self._verbose > 0: print(" loading model from '{}'".format(self._model_path))
                w2v[self._model_path] = loadGoogleW2v(self._model_path)
        self._model = w2v[self._model_path]
        def vectorize(sentence):
            words = sentence.split()
            words_vecs = [self._model[word] for word in words if word in self._model]
            if len(words_vecs) == 0:
                return self._model["hello"]
            words_vecs = np.array(words_vecs)
            return words_vecs.mean(axis=0)
        return np.array([vectorize(sentence) for sentence in X])

    def fit(self, X, y, **fitparma):
        return self

    def set_params(self, **params):
        self._model_path = params.get("model", None)
        self._verbose = params.get('verbose', 0)
        return self

    def get_params(self, deep=None):
        return {"model": self._model_path, "verbose": self._verbose}

    def save(self, fileout):
        if fileout is None: return
        if self._verbose > 0: print(" [{}] saving model to".format(str(self)), fileout)
        with open(fileout, 'a') as f:
            json.dump(self.get_params(), f)
    
    def summary(self):
        print("""{}(model={}, verbose={})""".format(str(self), str(self._model_path), str(self._verbose)))

    def __str__(self) -> str:
        return "Word2VectSA"

