import json
import os
import sys
import argparse
import time
import pandas as pd

from sentimental_hwglu.sa_random_forest import RandomForestTfid, RandomForestWord2Vec
from sentimental_hwglu.sa_svm import SVM_W2V, SVMTfidVec
from sentimental_hwglu.sa_naive import NaiveSA, NaiveSAPipeline
from sentimental_hwglu.sa_afinn import AFinnPipeline, VaderPipeline
from sentimental_hwglu.sa_logistic_regression import LogisticRegressionTfid, LogisticRegressionWord2Vec
from sentimental_hwglu.sa_lstm import LSTMPipeline
from sentimental_hwglu.sa_lstm_w2v import LSTMPipeline_W2V
import sentimental_hwglu.sa_model as sam
import pickle
import jdata as jd; ### lesen funktioniert nicht mit numpy

from sentimental_hwglu.utils import (
    Project,
    prepareIMDBdataset,
    loadIMDBdataset,
    prepareGoogleWords2Vec,
    get_timestamp
)
from sentimental_hwglu.plotter import (
    StatisticWordsParameters,
    plot_correlation_heatmap,
    plot_scores_grid_search, 
    plot_statistics_words, 
    plot_emoticons_heatmap,
    plot_venn_diagramm_for_words
)

from sentimental_hwglu.words_statistics import (
    WordStatistics
)

hwglu_commands = ['prepare', 'plot', 'words', 'experiment', 'grid_search', 'plot_grid_search']

def get_argparse():
    parser = argparse.ArgumentParser(description='sentimental_hwglu: command line tools to perform Sentimental Analysis')
    parser.prog = 'sentimental_hwlu'
    parser.add_argument('command')
    parser.add_argument('--out', '-o')
    parser.add_argument('--x_lim', nargs='+', default=None,type=float, )
    parser.add_argument('--y_lim', nargs='+', default=None, type=float, )
    parser.add_argument('--y_lim_tail',nargs='+', default=None, type=float, )
    parser.add_argument('--y_ticks_tail',default=3, type=int,   )
    parser.add_argument('--tail_pos',nargs='+', default=[.6, .4, .25, .25],type=float, )
    parser.add_argument('--tail', action='store_true', )
    parser.add_argument('--mode', action='store_true', )
    parser.add_argument('--mean', action='store_true', )
    parser.add_argument('--dimension', '-d', default=[],     nargs='+', type=str)
    parser.add_argument('--save', '-s')
    parser.add_argument('--models', '-m',nargs='+', type=str)
    parser.add_argument('--tests', '-t', default=[0.25], nargs='+', type=float, )
    parser.add_argument('--params', '-p')
    parser.add_argument('--timestamp',action='store_true')
    parser.add_argument('--verbose',  action='store_true', )
    return parser

class Dataset:
    def __init__(self, name):
        self.name = name

class GoogleWords2Vec(Dataset):
    def __init__(self, project: Project):
        self._project = project

    def prepare(self):
        prepareGoogleWords2Vec(project=self._project)

class IMDB(Dataset):
    def __init__(self, project: Project):
        self._project = project

    def prepare(self):
        prepareIMDBdataset(self._project)


    def plot(self, args):
        params = StatisticWordsParameters()
        dimension = args.dimension
        save = args.save
        if save is not None and len(save) > 0:
            params.set_save(True)
            params.set_outfile(save)
        if dimension == ['venn_words']:
            plot_venn_diagramm_for_words(params, args.out)
            return
        df = loadIMDBdataset(filename=self._project.csv_filename_extened)

        params.set_mode(args.mode)
        params.set_mean(args.mean)
        params.set_density(False)
        params.set_tail(args.tail)
        params.set_y_ticks_tails(args.y_ticks_tail)
        params.set_position_tail(args.tail_pos)

        if dimension == ['length'] or dimension == ['stamm_length']:
            params.set_xlim(args.x_lim)
            params.set_tail_ylim(args.y_lim_tail)
            params.set_xlabel('length words')
            params.set_ylabel('frequency')
            params.set_title("Distribution length of the Reviews")
            # params.set_position_tail([0.5, 0.28, 0.35, 0.35])
            params.set_hist_type('stepfilled')
            plot_statistics_words(params, df, dimension)
        elif dimension == ['words'] or dimension == ['stamm_words']:
            params.set_xlim(args.x_lim)
            params.set_tail_ylim(args.y_lim_tail)
            params.set_xlabel('number of words')
            params.set_ylabel('frequency')
            params.set_title("Distribution of the number of words in Reviews")
            # params.set_position_tail([0.5, 0.35, 0.35, 0.35])
            params.set_hist_type('stepfilled')
            plot_statistics_words(params, df, dimension)
        elif dimension == ['sentences'] or dimension == ['sentences']:
            params.set_xlim(args.x_lim)
            params.set_tail_ylim(args.y_lim_tail)
            params.set_xlabel('number of sentences')
            params.set_ylabel('frequency')
            params.set_title("Distribution number of sentences in Reviews")
            params.set_bins(95)
            params.set_hist_type('stepfilled')
            plot_statistics_words(params, df, 'sentences')
        elif dimension == ['emoticons']:
            plot_emoticons_heatmap(params, df)
        else:
            print(" try to plot using dimentions: ", dimension)
            params.set_xlim(args.x_lim)
            params.set_tail_ylim(args.y_lim_tail)
            params.set_xlabel('length words')
            params.set_ylabel('frequency')
            params.set_title("Distribution of {}".format(' '.join(dimension)))
            # params.set_position_tail([0.5, 0.28, 0.35, 0.35])
            params.set_hist_type('stepfilled')
            plot_statistics_words(params, df, dimension, plot_sentiment=False)

    def words(self, args):
        df = loadIMDBdataset(filename=self._project.csv_filename_extened)
        if args.save is None:
            raise RuntimeError("data path with -s")
        ws = WordStatistics(df, column=args.dimension[0])
        ws.createWordsSets()
        ws.dump(args.save)
        print(" saving all words")
        if not os.path.exists(self._project.all_words):
            df = loadIMDBdataset(filename=self._project.csv_filename_extened)
            columns = ['reviews', 'reviews_no_punctuation', 'reviews_no_stopwords', 'stamm', 'stamm_no_stopwords', 'stamm_no_punctuation', 'stamm_no_stop_punct']
            words = set()
            for c in columns:
                for tokens_list in df[c].str.split():
                    for w in tokens_list:
                        words.add(w)
            with open(self._project.all_words, 'w+') as f:
                for w in words:
                    f.write(w + '\n')

    def plot_grid_search(self, args):
        params = StatisticWordsParameters()
        # ======================================= #
        save = args.save
        if save is not None and len(save) > 0:
            params.set_save(True)
            params.set_outfile(save)
        # ======================================= #
        model_name = args.models[0]
        file_name = os.path.join(args.params, 'grid_search_{}.pkl'.format(model_name))
        with open(file_name, 'rb') as f:
            results = pickle.load(f)
        results_df = pd.DataFrame(results['cv_results'])
        
        d = args.dimension[0]
        params.set_xlim(args.x_lim)
        params.set_tail_ylim(args.y_lim)
        if d == 'scores':
            plot_scores_grid_search(params, results_df, model_name)
        elif d == 'correlation':
            plot_correlation_heatmap(params, results_df, model_name)
        elif d == 'hist':
            import sentimental_hwglu.plotter_grid_search as pgs 
            results_metrics = pgs.create_result_metrics(model_name, results_df)
            pgs.plot_metric_hist(params, model_name, results_metrics, y_lim=params.tail_ylim)
        else:
            print("ERROR !!!!")

    def _model_params_json2py_experiment(self, param):
        print("got params: ", param)
        if type(param) == dict:
            for k, v in param.items():
                if k.endswith('ngram_range'):
                    param[k] = (v[0], v[1])
        else: pass
        print("fixed params: ", param)
        return param

    def _model_params_json2py_grid_search(self, param):
        print("got params: ", param)
        if type(param) == dict:
            for k, v in param.items():
                if k.endswith('ngram_range'):
                    param[k] = [(x[0], x[1]) for x in v]
        else: pass
        print("fixed params: ", param)
        return param
    
    def grid_search(self, args):
        df = loadIMDBdataset(filename=self._project.csv_filename_extened)
        print(" |  running: ", ', '.join([str(m) for m in args.models]))
        results = {}
        for m in args.models:
            r = self.single_grid_search(m, df, args)
            results[m] = r
        return results

    def single_grid_search(self, m, df, args):
        print(" running model", str(m))
        out_dir = args.save
        if out_dir is not None:
            timestamp = ('_' + get_timestamp()) if args.timestamp else ''
            fileout_model = os.path.join(out_dir, 'grid_search_model{}'.format(timestamp))
            fileout = os.path.join(out_dir, 'grid_search_{}{}.pkl'.format(m, timestamp))
            fileout_js = os.path.join(out_dir, 'grid_search_{}{}.json'.format(m, timestamp))
            fileout_best_params = os.path.join(out_dir, 'best_params_{}{}.json'.format(m, timestamp))
            # with open(fileout, 'w+') as f: f.write('')
            # with open(fileout_js, 'w+') as f: f.write('')
        else:
            fileout_model = None
        # ==================================================================== #

        print(" running grid_search for model: {}".format(m))
        model = None
        if args.params is None or not os.path.exists(args.params):
            print(" no valid paramter file: got '{}'".format(args.params))
            return
        with open(args.params) as f:
            str_f = ''.join(f.readlines())
            param_json = json.loads(str_f)
            param_model = param_json['models'][m]
        param_grid = param_model['grid_search']
        param_grid = self._model_params_json2py_grid_search(param_grid)
        try: overwrite_params = param_model['common']
        except KeyError: overwrite_params = {}
        # param_model = param_model['parameters']
        params_opts = param_json['common']
        for k, v in overwrite_params.items(): params_opts[k] = v
        params_opts['dump_file_base'] = fileout_model
        if m == 'NaiveSA': model = NaiveSAPipeline()
        elif m == 'AFinn': model = AFinnPipeline()
        elif m == 'Vader': model = VaderPipeline()
        elif m == 'LogisticRegression': model = LogisticRegressionTfid()
        elif m == 'LogisticRegressionW2V': model = LogisticRegressionWord2Vec()
        elif m == 'RandomForest': model = RandomForestTfid()
        elif m == 'RandomForestW2V': model = RandomForestWord2Vec()
        elif m == 'SVM': model = SVMTfidVec()
        elif m == 'SVM_W2V': model = SVM_W2V()
        elif m == 'LSTM': model = LSTMPipeline()
        elif m == 'LSTM_W2V': model = LSTMPipeline_W2V()

        result = sam.run_grid_search(model, df, df.sentiment, param_grid=param_grid, params=sam.ExperimentParameters(**params_opts))
        if out_dir is not None:
            with open(fileout_best_params, 'w') as f: json.dump(result['best_params'], f)
            with open(fileout, 'wb') as f: pickle.dump(result, f)
            jd.save(result, fileout_js)
        return result

    def experiment(self, args):
        out_dir = args.save
        timestamp = get_timestamp()
        if out_dir is not None:
            fileout = os.path.join(out_dir, 'experment_{}.pkl'.format(timestamp))
            fileout_js = os.path.join(out_dir, 'experment_{}.json'.format(timestamp))
        df = loadIMDBdataset(filename=self._project.csv_filename_extened)
        print("running experiment for models: {}".format(', '.join(args.models)))
        models = {}
        for m in args.models:
            if m == 'NaiveSA': 
                models[m] = NaiveSAPipeline()
            elif m == 'AFinn': 
                models[m] = AFinnPipeline()
            elif m == 'Vader': 
                models[m] = VaderPipeline()
            elif m == 'SVM': 
                models[m] = SVMTfidVec()
            elif m == 'SVM_W2V': models[m] = SVM_W2V()
            elif m == 'LogisticRegression': 
                models[m] = LogisticRegressionTfid()
            elif m == 'LogisticRegressionW2V': 
                models[m] = LogisticRegressionWord2Vec()
            elif m == 'RandomForest': 
                models[m] = RandomForestTfid()
            elif m == 'RandomForestW2V': 
                models[m] = RandomForestWord2Vec()
            elif m == 'LSTM': models[m] = LSTMPipeline()
            elif m == 'LSTM_W2V': models[m] = LSTMPipeline_W2V()
            else: print(" no known model for '{}'".format(m))

        if args.params is None or not os.path.exists(args.params):
            print(" no valid paramter file: got '{}'".format(args.params))
            return
        param_json = {}
        with open(args.params) as f:
            str_f = ''.join(f.readlines())
            param_full_json = json.loads(str_f)
            param_json = param_full_json["models"]
            for _, p in param_json.items():
                self._model_params_json2py_experiment(p)
            try: param_cross_validation = param_full_json['cross_validation']
            except: param_cross_validation = {}

        params = {} 
        params["models_params"] = param_json
        for k, v in param_cross_validation.items():
            params[k] = v

        results = {}
        results['tests_splits'] = args.tests
        for t in args.tests:
            print(" running test: ", t)
            params['split_prec_tests'] = t
            params_test=sam.ExperimentParameters(**params)
            result = sam.run_experiments(models, df, df.sentiment, params=params_test)
            data = {}
            for d, v in result.items(): 
                data[d] = v.data()
            results[t] = data
            results[t]['config'] = params_test.get_parameters()
        if out_dir is not None:
            with open(fileout, 'wb') as f: 
                pickle.dump(results, f)
            jd.save(results, fileout_js)
        else:
            print(results)
        return results

def exec(args):
    parser = get_argparse()
    args = parser.parse_args(args)
    print("args: ", args)
    if args.out is None:
        print(" give a valid output directory")
        parser.print_help()
        sys.exit(1)
    project = Project(args.out)
    if args.dataset == 'imdb':
        dataset = IMDB(project)
    if args.dataset == 'googleW2v':
        dataset = GoogleWords2Vec(project)
        print("Invalid dataset name: ", args.dataset)
    if args.command == 'prepare': return dataset.prepare()
    if args.command == 'plot': return dataset.plot(args)
    if args.command == 'words': return dataset.words(args)
    if args.command == 'plot_grid_search': return dataset.plot_grid_search(args)
    if args.command == 'experiment': return dataset.experiment(args)
    if args.command == 'grid_search': return dataset.grid_search(args)

