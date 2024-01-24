import json
from sentimental_hwglu.sa_model import SentimentAnalysisPipeline
from sentimental_hwglu.utils import progressbar, logged_apply
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

import pandas as pd

from afinn import Afinn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# from https://nealcaren.org/lessons/wordlists/

class AFinnSA(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._afinn = Afinn(language='en')
    
    def fit(self, X, y, validateion_data=None):
        pass

    def predict(self, X):
        res = []
        max_n = len(X)
        for n, d in enumerate(X):
            progressbar(n, n_max=max_n, text='AFinn::predict:')
            res.append(1 if self._afinn.score(d) > 0 else 0)
        return res

    def set_params(self, **kwargs):
        self._params = kwargs
    
    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)

    def summary(self)-> str:
        return "AFinn: english"

    def save(self, fileout):
        if fileout is None: return
        if self._verbose > 0: print(" [{}] saving model to".format(str(self)), fileout)
        with open(fileout, 'a') as f:
            json.dump({"vocabulary": "Afinn:en"}, f)
    
    def __str__(self) -> str:
        return "AFinn"

class VaderSA(BaseEstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._analyzer = SentimentIntensityAnalyzer()
        import nltk 
        nltk.download('vader_lexicon')
    
    def fit(self, X, y, **fitparmas):
        pass

    def predict(self, X):
        sentiment = X.apply(self._analyzer.polarity_scores)
        sentiment_df = pd.DataFrame(sentiment.tolist())
        return logged_apply('Vader::predict:', sentiment_df['compound'], lambda x : 1 if x > 0 else 0)

    def set_params(self, **kwargs):
        self._params = kwargs

    def get_params(self, deep: bool = True) -> dict:
        return super().get_params(deep)

    def summary(self)-> str:
        return "Vader: SentimentIntensityAnalyzer"

    def save(self, fileout):
        if fileout is None: return
        if self._verbose > 0: print(" [{}] saving model to".format(str(self)), fileout)
        with open(fileout, 'a') as f:
            json.dump({"vocabulary": "VaderSA:en"}, f)

    def __str__(self) -> str:
        return "Vader"

class AFinnPipeline(SentimentAnalysisPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline.steps.append(('afinn', AFinnSA()))
        self._pipeline.set_params(**kwargs)

    def __str__(self) -> str:
        return "AFinnPipeline"

class VaderPipeline(SentimentAnalysisPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline.steps.append(('vader', VaderSA()))
        self._pipeline.set_params(**kwargs)

    def __str__(self) -> str:
        return "VaderPipeline"