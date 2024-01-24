from sklearn.base import BaseEstimator


class DataSelector(BaseEstimator):
    def __init__(self, **params):
        self.set_params(**params)

    def transform(self, X): 
        if self._verbose > 0: print(" [{}] selecting params: ".format(str(self)), self.get_params())
        try: 
            d = X[self._dimension]
        except: 
            d = X
        return d

    def fit(self, X, y, **fitparma):
        return self

    def set_params(self, **params):
        self._dimension = params.get('dimension', 'reviews')
        self._verbose = params.get('verbose', 0)
        return self

    def get_params(self, deep=None):
        return {"dimension": self._dimension, "verbose": self._verbose}

    def save(self, fileout):
        if fileout is None: return
        if self._verbose > 0: print(" [{}] saving model to".format(str(self)), fileout)
        with open(fileout, 'a') as f:
            json.dump(self.get_params(), f)
    
    def summary(self):
        print("""{}(dimension={}, verbose={})""".format(str(self), str(self._dimension), str(self._verbose)))

    def __str__(self) -> str:
        return "DataSelector"

