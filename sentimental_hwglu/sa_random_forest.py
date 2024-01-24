from sentimental_hwglu.sa_embeddings import Word2VectSA
from sentimental_hwglu.sa_model import SentimentAnalysisPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class RandomForestTfid(SentimentAnalysisPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline.steps.append(('vect', TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)))
        self._pipeline.steps.append(('rndf', RandomForestClassifier(random_state=self._random_state)))
        self._pipeline.set_params(**kwargs)

    def __str__(self) -> str:
        return "RandomForestTfid"

class RandomForestWord2Vec(SentimentAnalysisPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline.steps.append(('vect', Word2VectSA()))
        self._pipeline.steps.append(('rndf', RandomForestClassifier(random_state=self._random_state)))
        self._pipeline.set_params(**kwargs)

    def __str__(self) -> str:
        return "RandomForestW2Vec"