from sentimental_hwglu.sa_embeddings import Word2VectSA
from sentimental_hwglu.sa_model import SentimentAnalysisPipeline
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class LogisticRegressionTfid(SentimentAnalysisPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline.steps.append(('vect', TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)))
        self._pipeline.steps.append(('clf', LogisticRegression(random_state=self._random_state, solver='liblinear')))
        self._pipeline.set_params(**kwargs)

    def __str__(self) -> str:
        return "LogisticRegressionTfid"

class LogisticRegressionWord2Vec(SentimentAnalysisPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline.steps.append(('vect', Word2VectSA()))
        self._pipeline.steps.append(('clf', LogisticRegression(random_state=self._random_state, solver='liblinear')))
        self._pipeline.set_params(**kwargs)

    def __str__(self) -> str:
        return "LogisticRegressionW2Vec"