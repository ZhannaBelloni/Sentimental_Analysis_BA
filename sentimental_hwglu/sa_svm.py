from sentimental_hwglu.sa_embeddings import Word2VectSA
from sentimental_hwglu.sa_model import SentimentAnalysisPipeline
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler


class SVMTfidVec(SentimentAnalysisPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline.steps.append(('vect', TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)))
        # self._pipeline.steps.append(('max_abs', MaxAbsScaler()))
        # self._pipeline.steps.append(('std_scaler', StandardScaler(with_mean=False)))
        self._pipeline.steps.append(('svm', SVC(random_state=self._random_state)))
        self._pipeline.set_params(**kwargs)

    def __str__(self) -> str:
        return "SVM-Tfid"

class SVM_W2V(SentimentAnalysisPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._pipeline.steps.append(('vect', Word2VectSA()))
        # self._pipeline.steps.append(('max_abs', MaxAbsScaler()))
        # self._pipeline.steps.append(('std_scaler', StandardScaler(with_mean=False)))
        self._pipeline.steps.append(('svm', SVC(random_state=self._random_state)))
        self._pipeline.set_params(**kwargs)

    def __str__(self) -> str:
        return "SVM-W2V"
