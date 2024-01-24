import requests
import os
import tarfile
import re
import json
import random
import string
import shutil

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import gensim.downloader as api

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from gensim.models import KeyedVectors
from gensim import models
import time

# ================================================================================= #

def get_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S")


# ================================================================================= #
# Fetch the data
# ================================================================================= #

class Project:

    def __init__(self, outdir) -> None:
        self.url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        self.basedir = outdir
        self.outdir = os.path.join(self.basedir, 'datasets')
        self.outdir_zip = os.path.join(self.basedir, 'archives')
        self.dirdatabase = os.path.join(self.basedir, "datasets/aclImdb")
        self.dirdatawords = os.path.join(self.basedir, "datasets/words")
        self.dirdataGoogleW2v = os.path.join(self.basedir, "datasets/googleW2v")
        self.filename = "aclImdb_v1.tar.gz"
        self.csv_filename = os.path.join(self.outdir, "aclImdb_v1")
        self.csv_filename_clean = Project.get_filename(self.csv_filename, tag='clean')
        self.csv_filename_extened = Project.get_filename(self.csv_filename, tag='extended')
        self.all_words = os.path.join(self.basedir, 'datasets', 'all_words.csv')
        self.amazon_filename = os.path.join(self.outdir_zip, "All_Amazon_Review.json.gz")
        self.amazon_csv_filename = os.path.join(self.outdir, "All_Amazon_Review.csv")
        self.amazon_csv_filename_clean = os.path.join(self.outdir, "All_Amazon_Review_clean.csv")

        self.googleWords2vec_url = "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g"
        self.googleWords2vec_name = "word2vec-google-news-300.gz"
    
    def makeProjectStructure(self):
        try: os.mkdir(self.outdir)
        except: pass
        try: os.mkdir(self.outdir_zip)
        except: pass
        try: os.mkdir(self.dirdatabase)
        except: pass
        try: os.mkdir(self.dirdatawords)
        except: pass
        try: os.mkdir(self.dirdataGoogleW2v)
        except: pass

    @staticmethod 
    def get_filename(filename, tag=''):
        if tag == '':
            return filename + '.csv'
        else:
            return filename + '_' + tag + '.csv'


# ================================================================================= #
_positive = [' ' + x + ' ' for x in [
        ':-)', ':)', ':-]', ':]', ':-3', ':3', ':->', ':>', '8-)', '8)', 
        ':-}', ':}', ':o)', ':c)', ':^)', '=]', '=)', ':-D', ':D', '8-D', 
        '8D', 'x-D', 'xD;', 'X-D', 'XD', '=D', '=3', 'B^D' , ":'-)" , ":')",
        ':-*' , ':*', ':x', ';-)', ';)' , '*-)', '*)', ';-]', ';]', ';^)', 
        ':-,', ';D', ':-P', ':P', 'X-P', 'XP', 'x-p', 'xp', ':-p', ':p',
        ':-Þ', ':Þ', ':-þ', ':þ', ':-b', ':b', 'd:', '=p', '>:P', 'O:-)',
        'O:)', '0:-3', '0:3', '0:-)', '0:)', '0;^)', '|;-)', '<3', '\o/', 
        "D;", "D:", 
    ]]

_negative = [' ' + x + ' ' for x in [
        ":-(", ":(", ":-c", ":c", ":-<", ":<", ":-[", ":[", ":-||", ">:[",
        ":{", ":@", ">:(", ":'-(", ":'(", "D-':", "D:<", "D8",
        "D=", "DX", ":-O", ":O", ":-o", ":o", ":-0", "8-0", ">:O", "O_O",
        "o-o", "O_o", "o_O", "o_o", "O-O", ":-/", ":/", ":-.", ">:\\", ">:/",
        ":=/", "=\\", ":L", "=L", ":S", ":-|", ":|", ":$", ":-X", ":X", ":-#",
        ":#", ":-&", ":&", ">:-)", ">:)", "}:-)", "}:)", "3:-)", "3:)", ">;)",
        "|-O", "%-)", "%)", ":#", ":-#", "',:-|", "',:-l", "</3", ">.<",
    ]]

_CLEANR = re.compile('<.*?>') 
# _CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

class Emoticons:
    positive = _positive
    negative = _negative

# https://stackoverflow.com/questions/3160699/python-progress-bar
def progressbar(n, n_max, text='', step=100):
    if n % step == 0: 
        perc = 100.0 * n / n_max
        completed = int(perc * 20 / 100)
        remaining = 20 - completed
        print(" {} [{}{}] {:.1f}%".format(text, '#' * completed, ' ' * remaining, perc), end='\r')

def download_resource(file_url, outputdirectory, nameout):
    print(" downloading {}\n   to {} as {}".format(file_url, outputdirectory, nameout))
    outfile = os.path.join(outputdirectory, nameout)

    if os.path.exists(outfile):
        raise RuntimeError("File already downloaded")

    r = requests.get(file_url, stream = True)
    
    with open(outfile,"wb") as of:
        for chunk in r.iter_content(chunk_size=1024):  
             # writing one chunk at a time to pdf file
            if chunk:
                of.write(chunk)

def replace_emoticon(positive):
    if positive: return ' positive_emoticon '
    else:        return ' negative_emoticon '

def replace_emoticon_n(positive, n):
    if positive: return ' positive_emoticon' + '_' + str(n) + ' '
    else:        return ' negative_emoticon' + '_' + str(n) + ' '

def uncompress(filein, outdir):
    print(" uncompressing file {} to {}".format(filein, outdir))
    # open file
    with tarfile.open(filein) as file:
        # extracting file
        file.extractall(outdir)
        file.close()

def pandas2csv(df, fileout, shuffle=False, seed=0):
    if shuffle:
        np.random.seed(seed)
        df = df.reindex(np.random.permutation(df.index))
    df.to_csv(fileout, index=False)

def clean_spaces(text):
    text = re.sub(' +', ' ', text.lower())
    return text

def clean_html_tags(text):
    text = re.sub(_CLEANR, '', text).lower()
    text = re.sub(' +', ' ', text.lower())
    return text

def clear_emoticons(text):
    for n, e in enumerate(Emoticons.negative):
        text = text.replace(e, replace_emoticon(False))
    for n, e in enumerate(Emoticons.positive):
        text = text.replace(e, replace_emoticon(True))
    return text

def clear_emoticons_repeat(text, k):
    for _ in range(k):
        for n, e in enumerate(Emoticons.negative):
            text = text.replace(e, replace_emoticon(False))
        for n, e in enumerate(Emoticons.positive):
            text = text.replace(e, replace_emoticon(True))
    return text


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def extract_emoticons(text):
    # emoticons = re.findall(r"(?=("+'|'.join(Emoticons.negative_emoticons + Emoticons.positive_emoticons)+r"))", x)
    # emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    # return ' '.join(emoticons) if emoticons else None
    s = set()
    for e in Emoticons.negative + Emoticons.positive:
        if text.find(e) >= 0:
            s.add(e)
    return ''.join(sorted(list(s))) if s else None

def collection_emoticons_at_the_end_of_the_review(text):
    emoticons = extract_emoticons(text)
    text = re.sub('[\W]+', ' ', text.lower()) + emoticons
    return text

def loadIMDBdataset(filename):
    df = pd.read_csv(filename)
    return df

def replace_punctuation(x, remove=True):
    return clean_spaces(
        x.replace('?', '' if remove else ' ? ')
            .replace(':', '' if remove else ' : ')
            .replace(';', '' if remove else ' ; ')
            .replace(')', '' if remove else ' ) ')
            .replace('(', '' if remove else ' ( ')
            .replace('{', '' if remove else ' { ')
            .replace('}', '' if remove else ' } ')
            .replace('.', '' if remove else ' . ')
            .replace(',', '' if remove else ' , ')
            .replace('!', '' if remove else ' ! ')
            .replace('/', '' if remove else ' / ')
            .replace('"', '' if remove else ' " ')
            .replace(' \'', '' if remove else ' \' ')
            .replace('\' ', '' if remove else ' \' ')
    )

def isolate_punctuation(x):
    return replace_punctuation(x, remove=False)

def remove_punctuation(df, src='reviews', dst= 'reviews_no_punctuation'):
    print("   * remove punctuation "); df[dst]   = df[src].apply(replace_punctuation)
    return df

# https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
def logged_apply(text, g, func, *args, **kwargs):
    step_percentage = 100. / len(g)
    import sys
    sys.stdout.write('   {}:   0%'.format(text))
    sys.stdout.flush()

    def logging_decorator(func):
        def wrapper(*args, **kwargs):
            progress = wrapper.count * step_percentage
            sys.stdout.write('\033[D \033[D' * 4 + format(progress, '3.0f') + '%')
            sys.stdout.flush()
            wrapper.count += 1
            return func(*args, **kwargs)
        wrapper.count = 0
        return wrapper

    logged_func = logging_decorator(func)
    res = g.apply(logged_func, *args, **kwargs)
    sys.stdout.write('\033[D \033[D' * 4 + format(100., '3.0f') + '%' + '\n')
    sys.stdout.flush()
    return res

def remove_stopwords(df, src='reviews_no_punctuation', dst='reviews_no_stopwords'):
    print("   * remove stopwords   "); 
    # df[dst]   = df[src].apply(lambda x : ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
    df[dst] = logged_apply('* remove stopwords', df[src], lambda x : ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
    return df

def apply_stammer(df, src='reviews', dst='stamm'):
    print("   * remove stopwords   "); 
    # df[dst]   = df[src].apply(lambda x : ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
    df[dst] = logged_apply('* apply stammer:', df[src], lambda x : ' '.join([word for word in tokenizer_porter(x) if word not in stopwords.words('english')]))
    return df

def apply_stammer_no_stop_words(df, src='reviews_no_stopwords', dst='stamm_no_stopwords'):
    print("   * remove stopwords   "); 
    # df[dst]   = df[src].apply(lambda x : ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
    df[dst] = logged_apply('* stammer no stopwodrds:', df[src], lambda x : ' '.join([word for word in tokenizer_porter(x)]))
    return df

def imdb_clean_frame(df):
    print("   * extract emoticons  "); df['emoticons'] = df.reviews.apply(extract_emoticons)
    print("   * clear emoticons    "); df['reviews']   = df.apply(lambda x: clear_emoticons_repeat(x.reviews, len(x.emoticons.split()) if x.emoticons is not None else 0), axis=1)
    print("   * punctuation        "); df['reviews']   = df.reviews.apply(isolate_punctuation)
    print("   * clean html tags    "); df['reviews']   = df.reviews.apply(clean_html_tags)
    print("   * remove punctuation "); df['reviews_no_punctuation'] = df.reviews.apply(replace_punctuation)
    df = remove_stopwords(df)
    return df

def imdb_clean_stamm(df):
    print("   * remove punctuation "); df['stamm_no_punctuation'] = df.stamm.apply(replace_punctuation)
    print("   * . for no_stopwords "); df['stamm_no_stop_punct'] = df.stamm_no_stopwords.apply(replace_punctuation)
    return df

def aclImdb2csv(dirin, outfile_base, seed=0):
    imdb_file_csv = Project.get_filename(outfile_base)
    print(" preparing csv {} file for IMDb\n in directory {}\n seed = {}".format(imdb_file_csv, dirin, seed))
    if not os.path.exists(imdb_file_csv):
        labels = {"pos": 1, "neg": 0}
        df = pd.DataFrame()
        number_files = 0
        for s in ('test', 'train'):
            for l in ('pos', 'neg'):
                path = os.path.join(dirin, s, l)
                for file in os.listdir(path):
                    number_files += 1
        n = 0
        for s in ('test', 'train'):
            for l in ('pos', 'neg'):
                path = os.path.join(dirin, s, l)
                for file in os.listdir(path):
                    if n % 100 == 0: 
                        perc = 100.0 * n / number_files
                        completed = int(perc * 20 / 100)
                        remaining = 20 - completed
                        print(" reading files [{}{}] {:.1f}%".format('#' * completed, ' ' * remaining, perc), end='\r')
                    with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                        n += 1
                        text = infile.read()
                    df = pd.concat([df, pd.DataFrame([[text, labels[l]]])], ignore_index=True)
        df.columns = ['reviews', 'sentiment']
        print(" creating file: {}".format(imdb_file_csv))
        np.random.seed(seed)
        df = df.reindex(np.random.permutation(df.index))
        df.to_csv(imdb_file_csv, index=False)
    
    imdb_extended_file_csv = Project.get_filename(outfile_base, tag='extended')
    if not os.path.exists(imdb_extended_file_csv):
        df = df if df is not None else loadIMDBdataset(imdb_file_csv)
        n_reviews = len(df.reviews)
        print(" creating file: {}".format(imdb_extended_file_csv))
        print(" --------------- ")
        print("   * backup original  "); df['original']  = df.reviews
        df = imdb_clean_frame(df)
        print("   * calculate length "); df['length']    = df.reviews.apply(lambda x : len(x))
        print("   * calculate words  "); df['words']     = df.reviews.apply(lambda x : len(x.split()))
        print("   * calc. sentences  "); df['sentences'] = df.reviews.apply(lambda x : len([w for w in re.split('\.|!|\?|:', x) if len(w.strip()) > 0 and len(w.split()) > 3]))
        print(" --------------- ")
        print("   * positive_emoji   "); df['positive_emoticons'] = df.reviews.apply(lambda x : len(re.findall('positive_emoticon', x)))
        print("   * negative_emoji   "); df['negative_emoticons'] = df.reviews.apply(lambda x : len(re.findall('negative_emoticon', x)))
        print(" --------------- ")
        print(" using stammer ")
        df = apply_stammer(df)
        df = apply_stammer_no_stop_words(df)
        df = imdb_clean_stamm(df)
        print("   * calculate length "); df['stamm_length'] = df.stamm.apply(lambda x : len(x))
        print("   * calculate words  "); df['stamm_words'] = df.stamm.apply(lambda x : len(x.split()))
        print("   * calc. sentences  "); df['stamm_sentences'] = df.stamm.apply(lambda x : len([w for w in re.split('\.|!|\?|:', x) if len(w.strip()) > 0 and len(w.split()) > 3]))

        pandas2csv(df, imdb_extended_file_csv)

    # === add emoticons ===
    return
    # df = df if df is not None else loadIMDBdataset(imdb_extended_file_csv)
    # pandas2csv(df, imdb_extended_file_csv)
    # =================================================== #
    # =================================================== #
    imdb_file_csv = Project.get_filename(outfile_base, tag='clean')
    print(" creating file: {}".format(imdb_file_csv))
    df.reviews = df.reviews.apply(clean_html_tags)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv(imdb_file_csv, index=False)
    # =================================================== #
    imdb_file_csv = Project.get_filename(outfile_base, tag='clean_emoticons')
    print(" creating file: {}".format(imdb_file_csv))
    df.reviews = df.reviews.apply(collection_emoticons_at_the_end_of_the_review)
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv(imdb_file_csv, index=False)
    # =================================================== #

def prepareIMDBdataset(project: Project):
    project.makeProjectStructure()
    try:
        download_resource(project.url, project.outdir_zip, project.filename)
        uncompress(os.path.join(project.outdir_zip, project.filename), project.outdir)
    except Exception as e: print(" " + str(e))
    try:
        aclImdb2csv(project.dirdatabase, project.csv_filename)
    except Exception as e: print(" " + str(e))

def prepareGoogleWords2Vec(project: Project):
    project.makeProjectStructure()
    try:
        dst_path = os.path.join(project.dirdataGoogleW2v, project.googleWords2vec_name)
        if os.path.exists(dst_path):
            raise RuntimeError("File '" + dst_path + "' already downloaded")
        wv = api.load('word2vec-google-news-300', return_path=True)
        print(" file downloaded into ", wv)
        print(" moving file into ", dst_path)
        shutil.move(wv, dst_path)
    except Exception as e: print(" " + str(e))

def loadGoogleW2v(project_or_path):
    if type(project_or_path) == Project:
        word2vec_path = os.path.join(project_or_path.dirdataGoogleW2v, project_or_path.googleWords2vec_name)
    else:
        word2vec_path = project_or_path
    w2v_model = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    return w2v_model

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

#### Fetch the data
def loadJsonAmazon2PandaFrame(filein, n_max=-1, nameout=None, verbose=False):
    fileout = nameout if nameout is not None else filein.replace('json', 'csv')
    df = {}
    if n_max > 0:
        tot = n_max
    else:
        with open(filein, "rbU") as f:
            n_max = sum(1 for _ in f)
    with open(filein) as f:
       for n, line in enumerate(f):
          if verbose and n_max > 0 and (n % 1000) == 0: print(" processing {} / {} ({} %)".format(n, tot, n * 100 // tot), end='\r')
          if n_max > 0 and n > n_max: break
          df[n] = json.loads(line)
    if verbose: print(" creating DataFrame)")
    df = pd.DataFrame.from_dict(df, orient='index')
    df['sentiment'] = df.overall.apply(lambda x : 1 if x <=3 else 0)
    if verbose: print(" writing data into {})".format(fileout))
    df.to_csv(fileout, columns=['sentiment', 'reviewText'], index_label=False)
    return df

def loadAmazonDataset(filename):
    df = pd.read_csv(filename)
    return df

def test_naive(X_train, y_train, X_test, y_test, tk, name, tk_name, X=None, y=None, model=None):
    if model is None:
        return 0
    naive_sa = model(verbose=True)
    naive_sa.fit(X_train, y_train)
    score = naive_sa.score(X_test, y_test)
    print(" Score of Native SA test: ", score)
    if X is not None:
        score_all = naive_sa.score(X, y)
        print(" Score of Native SA all:  ", score_all)
    else:
        score_all = 0
    return score, score_all

def test_FtidfVectorier(X_train, y_train, X_test, y_test, tk, name, tk_name, X=None, y=None, model=TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)):
    stop = stopwords.words('english')
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

    params = {
            'vect__ngram_range': (1, 1),
            'vect__stop_words':  stop,
            'vect__tokenizer':   tk,
            'clf__penalty':      'l2',
            'clf__C':            10.0,
        }

    lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
    lr_tfidf.set_params(**params)
    print(" fitting train data")
    lr_tfidf.fit(X_train, y_train)
    print(" calculating scores on test data")
    score = lr_tfidf.score(X_test, y_test)
    print("[%s_%s] Korrektklassifizierungsrate Test: %.3f" % (name, tk_name, score))
    if X is not None:
        score_all = lr_tfidf.score(X, y)
        print("[%s_%s:all] Korrektklassifizierungsrate Test: %.3f" % (name, tk_name, score_all))
    else:
        score_all = 0
    return score, score_all