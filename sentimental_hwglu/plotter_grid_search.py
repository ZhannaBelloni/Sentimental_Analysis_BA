import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib_venn import venn2
import pandas as pd
import numpy as np
from scipy import stats
# import seaborn as sn
import re
import pickle
from sentimental_hwglu.plotter import StatisticWordsParameters


def load_results_df(output_dir, model_name):
    file_name = os.path.join(output_dir, 'grid_search_{}.pkl'.format(model_name))
    if model_name == 'LogisticRegressionW2V':
        file_name = os.path.join(output_dir, 'grid_search_{}_cv_results.pkl'.format(model_name))
    with open(file_name, 'rb') as f:
        results = pickle.load(f)
    if model_name == 'LogisticRegressionW2V':
        results_df = pd.DataFrame(results)
    else:
        results_df = pd.DataFrame(results['cv_results'])
    return results_df

def create_result_metrics(model_name, results_df):
    w2v_path="/data//datasets/googleW2v/word2vec-google-news-300.gz"
    try:
        metrics = ['accuracy', 'f1', 'recall', 'precision']
        metric = metrics[0]
        results_df = results_df.sort_values(by=["rank_test_{}".format(metric)])
    except:
        metrics = ['score']
        metric = metrics[0]
        results_df = results_df.sort_values(by=["rank_test_{}".format(metric)])
    if model_name.startswith("RandomForest"):
        print(model_name)
        results_df = results_df.set_index(results_df["params"].apply(
            lambda x: "_".join(
                str(val).replace(w2v_path, '')
                        .replace('1', '')
                        .replace('5', '') 
                        .replace('reviews_no_stopwords', 'R.noS.') 
                        .replace('reviews_no_punctuation', 'R.noP.') 
                        .replace('stamm_no_punctuation', 'S.noP.') 
                        .replace('stamm_no_stop_punct', 'S.noS.P.') 
                        for val in x.values()))).rename_axis("hyperparam")
    if model_name.startswith("LogisticReg"):
        print(model_name)
        results_df = results_df.set_index(results_df["params"].apply(
            lambda x: '{' + ",".join([v.replace('_', '') for v in
                [str(val).replace(w2v_path, '')
                        .replace('None', '') 
                        .replace('True', '') 
                        .replace('l2', '') 
                        .replace('reviews_no_stopwords', 'R.noS.') 
                        .replace('reviews_no_punctuation', 'R.noP.') 
                        .replace('stamm_no_punctuation', 'S.noP.') 
                        .replace('stamm_no_stop_punct', 'S.noS.P.') 
                        for val in x.values()]])
        .replace(',,', '').replace('_1_', ', ').replace('.0', '.0, ') + '}'
                        )).rename_axis("hyperparam")
    else:
        results_df = results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val).replace(w2v_path, '') for val in x.values()))).rename_axis("kernel")
    # results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]]
    columns = [
        "mean_test_{}",
        # "std_test_{}",
        ]
    projec = ['rank_test_{}'.format(metric)]
    for m in metrics:
        for c in columns:
            projec.append(c.format(m))
    results_df[
        ["rank_test_{}".format(metric), 
        "mean_test_{}".format(metric), 
        # "std_test_{}".format(metric)
        ]
        ]

    results_metrics = results_df[projec]
    list_columns = results_metrics.columns
    map_rename = {}
    for c in list_columns:
        map_rename[c] = c.replace('rank_test_accuracy', 'rank').replace('_test_', ' ')
    print(map_rename)

    results_metrics = results_metrics.rename(columns=map_rename)
    return results_metrics

def get_model_scores(results_df, metric):
    model_scores = results_df.filter(regex=r"split\d*_test_{}".format(metric))
    return model_scores


def plot_metric_hist(params: StatisticWordsParameters, model_name, results_metrics, y_lim=None):

    # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    values_to_plot = results_metrics
    if y_lim is None:
        y_lim = [0.8, 1.0]
    fig, ax = plt.subplots()

    accuracy = values_to_plot['mean accuracy'].values
    f1 = values_to_plot['mean f1'].values
    recall = values_to_plot['mean recall'].values
    precision = values_to_plot['mean precision'].values

    barWidth = 0.2
    # Set position of bar on X axis 
    br1 = np.arange(len(accuracy)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    br4 = [x + barWidth for x in br3] 
    experiments = values_to_plot.index

    # Make the plot
    ax.set_title("Histogramm verschiedener Metriken\n für das Modell {}.".format(model_name), fontsize=14)
    ax.bar(br1, accuracy, width = barWidth, 
            edgecolor ='k', label ='ACC') 
    ax.bar(br2, f1, width = barWidth, 
            edgecolor ='k', label ='F1') 
    ax.bar(br3, recall, width = barWidth, 
            edgecolor ='k', label ='recall') 
    ax.bar(br4, precision, width = barWidth, 
            edgecolor ='k', label ='precision') 

    
    # Adding Xticks 
    ax.set_xlabel('Hyperparameters') #, fontweight ='bold', fontsize = 15) 
    ax.set_ylabel('metric value') #, fontweight ='bold', fontsize = 15) 
    ax.set_xticks([r + barWidth for r in range(len(precision))], 
            experiments, rotation=90)
    ax.set_ylim(y_lim)
    ax.grid('minor')
    
    ax.legend(ncol=4, loc='upper right')

    if params.save:
        print(" saving file to {}".format(params.outfile))
        plt.savefig(params.outfile)
    else:
        plt.show()

def plots_cv_metrics(results_df, metric, max_=30):
    import seaborn as sns
    # create df of model scores ordered by performance
    model_scores = get_model_scores(metric)

    # plot 30 examples of dependency between cv fold and AUC scores
    fig, ax = plt.subplots()
    sns.lineplot(
        data=model_scores.transpose().iloc[:max_],
        dashes=False,
        palette="Set1",
        marker="o",
        alpha=0.5,
        ax=ax,
    )
    ax.grid('minor')
    ax.set_xlabel("CV test fold", size=12, labelpad=10)
    ax.set_ylabel("Model accuracy", size=12)
    ax.tick_params(bottom=True, labelbottom=False)
    plt.legend([])
    plt.show()

def plot_correlation_heat_map():
    pass