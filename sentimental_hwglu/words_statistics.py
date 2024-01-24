import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from sentimental_hwglu.utils import progressbar

def my_timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f' function executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

class WordStatisticsVect:

    def __init__(self) -> None:
        self._vect = None
        self._bag = None
        self._agg = None

    def words2Vect(self, X, vectorizer, tokenizer = None):
        self._vect = vectorizer(tokenizer=tokenizer)
        self._bag = self._vect.fit_transform(X).toarray()
        self._agg = self._bag.sum(axis=0)

    def _reshape(self, Z):
        return np.transpose(Z.reshape(Z.shape[1]))

    def words2VectFiltered(self, X, y, vectorizer, tokenizer = None, filter=None):
        data = X.loc[y.apply(filter)] if filter is not None else X
        self._vect = vectorizer(tokenizer=tokenizer)
        self._bag = self._vect.fit_transform(data).toarray()
        self._agg = self._bag.sum(axis=0)

    def hist(self, sort=False):
        if sort: plt.plot(np.sort(self._agg))
        else: plt.plot(self._agg)
        plt.show()

    def top(self, n=10):
        # indexes = np.argpartition(self._agg, n)[-n:]
        indexes = np.argsort(-self._agg)[:n]
        names = self._vect.get_feature_names_out()
        top_words = [names[k] for k in indexes]
        return top_words

    def getAggregated(self):
        return self._agg

    def getWords(self):
        return self._vect.get_feature_names_out()


class WordStatistics:

    index_neg = 0
    index_pos = 1
    filter_pos = lambda x : x == WordStatistics.index_pos
    filter_neg = lambda x : x == WordStatistics.index_neg

    def __init__(self, df, column='reviews', X=None, y=None) -> None:
        self._tag_review = column
        self._tag_sentitment = "sentiment"
        if df is not None: 
            self._X = df[self._tag_review]
            self._y = df[self._tag_sentitment]
        else:
            self._X = X
            self._y = y
        self._positive_words = {}
        self._negative_words = {}
        self._common_words = {}
        self._only_positive_words = set()
        self._only_negative_words = set()
        self._df = None

    def _reset_words_sets(self):
        self._positive_words = {}
        self._negative_words = {}
        self._common_words = {}
        self._only_positive_words = set()
        self._only_negative_words = set()

    @my_timer_func
    def _create_common_words(self):
        print(" create common words")
        words_pos = set(self._positive_words.keys())
        words_neg = set(self._negative_words.keys())
        words_common = words_pos.intersection(words_neg)
        for w in words_common:
            tmp = [0] * 2
            tmp[WordStatistics.index_neg] = self._negative_words[w]
            tmp[WordStatistics.index_pos] = self._positive_words[w]
            self._common_words[w] = tmp

    @my_timer_func
    def _create_only_positive_words(self):
        print(" create only positive words")
        words_pos = set(self._positive_words.keys())
        words_neg = set(self._negative_words.keys())
        self._only_positive_words = words_pos.difference(words_neg)
        _only_negative_words = words_pos.difference(words_neg)
        self._only_positive_words = {}
        for k in _only_negative_words:
            try: self._only_positive_words[k] = self._positive_words[k]
            except: 
                print("word ", k, " not found in POSITIVE words!!")
                break

    @my_timer_func
    def _create_only_negative_words(self):
        print(" create only negative words")
        words_pos = set(self._positive_words.keys())
        words_neg = set(self._negative_words.keys())
        _only_negative_words = words_neg.difference(words_pos)
        self._only_negative_words = {}
        for k in _only_negative_words:
            try: self._only_negative_words[k] = self._negative_words[k]
            except: 
                print("word ", k, " not found in NEGATIVE words!!")
                break
    
    @my_timer_func
    def createWordsSets(self):
        X_pos = self._X.loc[self._y.apply(WordStatistics.filter_pos)]
        X_neg = self._X.loc[self._y.apply(WordStatistics.filter_neg)]
        self._reset_words_sets()
        k = 0
        for set_words, reviews in [(self.positiveWords, X_pos), (self.negativeWords, X_neg)]:
            k += 1
            print(" running set_words ", k, "/ 2")
            t0 = time.time()    
            tot = len(reviews)
            for n, review in enumerate(reviews):
                progressbar(n, tot, 'tokenization for review ', 1000)
                for tk in review.split():
                    if len(tk.strip()) <= 1: continue
                    try: set_words[tk] += 1
                    except KeyError: set_words[tk] = 1
            progressbar(n, tot, 'tokenization for review ', 1)
            t1 = time.time()    
            print("\n tokenization took ", t1 - t0, " sec.")
        self._create_common_words()
        self._create_only_negative_words()
        self._create_only_positive_words()
        
    @property
    def positiveWords(self): return self._positive_words
    @property
    def negativeWords(self): return self._negative_words
    @property
    def commonWords(self): return self._common_words
    @property
    def onlyPositive(self): return self._only_positive_words
    @property
    def onlyNegative(self): return self._only_negative_words

    def commonWordsDataFrame(self):
        if self._df is not None: return self._df
        words, count_use, count_positive, count_negative, fraction_positive, fraction_negative = [], [], [], [], [], []
        for w, f in self._common_words.items():
            p, n = f[1], f[0]
            words.append(w) 
            count_use.append(p + n)
            count_positive.append(p)
            count_negative.append(n)
            fraction_positive.append(p / (n + p))
            fraction_negative.append(n / (n + p))
        self._df = pd.DataFrame({
            'word': words, 
            'count_use': count_use,
            'count_positive': count_positive,
            'count_negative': count_negative,
            'fraction_positive': fraction_positive,
            'fraction_negative': fraction_negative
            })
        return self._df

    def dump(self, filenamebase):
        _do_write_neg = lambda x : [0, x]
        _do_write_pos = lambda x : [x, 0]
        _do_write_np = lambda x : [x[1], x[0]]
        for suff, words, writer in [
            ('_pos',      self._positive_words,      _do_write_pos),
            ('_neg',      self._negative_words,      _do_write_neg),
            ('_only_pos', self._only_positive_words, _do_write_pos),
            ('_only_neg', self._only_negative_words, _do_write_neg),
            ('_common',   self._common_words,        _do_write_np),
        ]:
            filename = filenamebase + suff + '.csv'
            with open(filename, 'w+', encoding="utf-8") as f:
                print(" writing file: ", filename)
                f.write('word;count_use;count_positive;count_negative;fraction_positive;fraction_negative\n')
                for w, count in words.items():
                    p, n = writer(count)
                    f.write(
                        w + ';' 
                        + str(p + n) + ';'
                        + str(p) + ';'
                        + str(n) + ';'
                        + str(p / (n + p)) + ';'
                        + str(n / (n + p)) + ';'
                        + '\n')
