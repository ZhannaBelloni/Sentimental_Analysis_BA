from abc import ABC, abstractmethod
import pandas as pd
from sentimental_hwglu.sa_data_selector import DataSelector
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
import pickle

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

class SentimentAnalysisPipeline(ClassifierMixin):
    def __init__(self, **kwargs):
        self._params = {}
        self._pipeline = Pipeline([("data_selector", DataSelector())])
        self._random_state = kwargs.get("random_state")
        self._dimension = kwargs.get("dimension", 'reviews')
        self._history = None
    
    def _get_data(self, X):
        try: X = X[self._dimension]
        except: pass
        return X
    
    def get_history(self):
        return self._history

    def fit(self, X, y, **fitparmas):
        if self._pipeline is None:
            raise RuntimeError("Invalid pipeline !!!!")
        self._pipeline.fit(X, y)

    @property
    def classes_(self):
        return np.array([1, 0])

    def get_params(self, deep=None): 
        if self._pipeline is None:
            return {}
        return self._pipeline.get_params(deep)

    def predict(self, X, **predict_params): 
        if self._pipeline is None:
            raise RuntimeError("Invalid pipeline !!!!")
        return self._pipeline.predict(X, **predict_params)

    def predict_proba(self, X, **predict_proba_params): 
        if self._pipeline is None:
            raise RuntimeError("Invalid pipeline !!!!")
        return self._pipeline.predict_proba(X, **predict_proba_params)

    def score(self, X, y): 
        if self._pipeline is None:
            raise RuntimeError("Invalid pipeline !!!!")
        return self._pipeline.score(X, y)

    def set_params(self, **kwargs): 
        self._params = kwargs
        if self._pipeline is not None:
            return self._pipeline.set_params(**kwargs)
        return self

    def transform(self, X): 
        if self._pipeline is None:
            raise RuntimeError("Invalid pipeline !!!!")
        return self._pipeline.transform(X)

    # def save(self, fileout):
    #     print("WARNING overwrite save to save the model in used.")
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


    def load(self, filein):
        if self._pipeline is None:
            raise RuntimeError("Invalid pipeline !!!!")
        return self._pipeline.load(filein)
    
    def __str__(self) -> str:
        return "None"

    def summary(self)-> str:
        for p in self._pipeline:
            try: p.summary()
            except: print(p)


class SA_Result():
    def __init__(self, precision=0.0, recall=0.0, scores=None, params=None) -> None:
        self._precision = precision
        self._recall = recall
        self._scores = scores
        self._params = params
    def setScores(self, s):
        self._scores = s
    def scores(self):
        return self._scores
    def setPrecision(self, p):
        self._precision = p
        return self
    def setRecall(self, r):
        self._recall = r
        return self
    def precision(self):
        return self._precision
    def recall(self):
        return self._recall
    def f1_score(self):
        denominator = self._precision + self._recall
        if denominator == 0: return 0.0
        else: return 2 * (self._precision * self._recall) / denominator
    def set_parameter(self, p):
        self._params = p
    def __str__(self):
        return str(self.data())
    def data(self):
        return {
            "recall": self.recall(),
            "precision": self.precision(),
            "f1": self.f1_score(),
            "scores": self._scores if self._scores else [],
            "params": self._params if self._params else {}
        }

class ExperimentParameters:
    def __init__(self, **kwargs):
        self.split_prec_tests = kwargs.get('split_prec_tests')
        self.cv = int(kwargs.get('cv', 2))
        self.n_jobs = int(kwargs.get('n_jobs', 1))
        self.scoring = kwargs.get("scoring", "accuracy")
        self.verbose = kwargs.get("verbose", 0)
        self.dump_file_base = kwargs.get("dump_file_base", None)
        self.n_repetition = kwargs.get("n_repetition", 1)
        self.n_jobs = kwargs.get("n_jobs", 1)
        self._params = kwargs
    
    def get_parameters(self):
        return self._params
    
    def get_params_for_models(self, model_name:str):
        try: return self._params["models_params"][model_name]
        except KeyError as e: print(" getting params for models failed: ", e); raise e

def run_sentimental_analysis_pipeline(pipeline: SentimentAnalysisPipeline, X_train, y_train, X_test, y_test, **parameters):
    pipeline.set_params(**parameters)
    pipeline.summary()
    pipeline.fit(X_train, y_train, validation_data=(X_test, y_test))
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return SA_Result(
        precision=precision_score(y_true=y_test, y_pred=y_pred), 
        recall=recall_score(y_true=y_test, y_pred=y_pred)
    )

def run_sa_cross_validation_pipeline(
        pipeline: SentimentAnalysisPipeline, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        experiment_params: ExperimentParameters=ExperimentParameters(),
        **parameters):
    # pipeline.set_params(**parameters)
    # pipeline.set_params(**experiment_params.get_params_for_models(name))
    pipeline.summary()
    scores = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=experiment_params.cv,
        n_jobs=experiment_params.n_jobs,
        scoring=experiment_params.scoring
    )
    y_pred = cross_val_predict(
        estimator=pipeline,
        X=X_test,
        y=y_test,
        cv=10,
    ) 
    print(scores)
    print(y_preds)
    return SA_Result(
        precision=precision_score(y_true=y_test, y_pred=y_pred), 
        recall=recall_score(y_true=y_test, y_pred=y_pred),
        scores=scores
    )

def run_experiments(models, X, y, params: ExperimentParameters=ExperimentParameters()):
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.split_prec_tests, random_state=42)
    for name, model in models.items():
        print(" Running pipeline: " + str(model) + ", test_size: " + str(params.split_prec_tests))
        st = time.time()
        model.set_params(**params.get_params_for_models(name))
        # r = run_sentimental_analysis_pipeline(model, X_train, y_train, X_test, y_test)
        res = run_sa_cross_validation_pipeline(model, X_train, y_train, X_test, y_test, experiment_params=params)
        res.set_parameter(params.get_params_for_models(name))
        et = time.time()
        results[str(model)] = res
        print(" precision = {}, recall = {}, F1 = {}".format(res.precision(), res.recall(), res.f1_score()))
        print(' Execution time:', et - st, 'seconds')
    for k, r in results.items():
        print("------------------------------------")
        print(k, ":", r)
        print("    precision: ", r.precision())
        print("    recall   : ", r.recall())
        print("    F1       : ", r.f1_score())
        print("    data     : ", r.data())
    return results

# === = === = === = === = === = === = === = === = === = === = === #
# model.set_params(**params.get_parameters())
# model.summary()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params.split_prec_tests, random_state=42)
# === = === = === = === = === = === = === = === = === = === = === #
def run_grid_search(model, X, y, param_grid, params: ExperimentParameters=ExperimentParameters()):
    st = time.time()
    params_dict = params.get_parameters()
    print(" model default parameters: ", params_dict)
    print(" grid_search parameters:   ", param_grid)
    X_train, y_train = X, y
    # https://stackoverflow.com/questions/44636370/scikit-learn-gridsearchcv-without-cross-validation-unsupervised-learning
    n_train = 1
    n_test = 1
    if params.n_repetition == 1 or params.cv == 1:
        cv = params.cv if params.cv > 1 else [(slice(None), slice(None))]
    else:
        cv = RepeatedStratifiedKFold(n_splits=params.cv, n_repeats=params.n_repetition)
        n_train = len(list(cv.split(X, y))[0][0])
        n_test = len(list(cv.split(X, y))[0][1])
    gs = GridSearchCV(
        estimator=model, 
        param_grid=param_grid, 
        scoring=params.scoring, 
        cv=cv, 
        verbose=params_dict.get('verbose', 0) , 
        error_score='raise', 
        refit="accuracy",
        n_jobs=params.n_jobs
        )
    print(" Running grid_search: " + str(model))
    gs.fit(X_train, y_train)
    print(" best estimator: ", gs.best_estimator_)
    print(" best params:    ", gs.best_params_)
    print(" best score:     ", gs.best_score_)
    print(" cv_results_:\n", pd.DataFrame(gs.cv_results_))
    et = time.time()
    print(' Execution time:', et - st, 'seconds')
    if params.dump_file_base is not None:
        filename = params.dump_file_base + '_' + str(model) + '.pkl'
        with open(filename, 'wb+') as f:
            pickle.dump(gs.best_estimator_, f)
    return {
        "best_estimator": params.dump_file_base,
        "best_score": gs.best_score_,
        "best_params": gs.best_params_,
        "cv_results": gs.cv_results_,
        "default_params": params.get_parameters(),
        "params_grid": param_grid,
        "execution_time": et - st,
        "grid_search": gs,
        "n_train": n_train,
        "n_test": n_test
    }

