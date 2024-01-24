import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib_venn import venn2
import pandas as pd
import numpy as np
from scipy import stats
# import seaborn as sn
import re
font = {
    'size'   : 16,
    'family': 'serif'
    }
matplotlib.rc('font', **font)

class StatisticWordsParameters:
    def __init__(self) -> None:
        self.save = False
        self.x_lim = None
        self.bins = None
        self.title = ''
        self.x_label = None
        self.y_label = None
        self.tail = False
        self.density = False
        self.tail_ylim = None
        self.position_tail = [.6, .4, .25, .25]
        self.hist_type = 'stepfilled'
        self.mode = False
        self.mean = False
        self.outfile = None
        self.y_ticks_tail = 3

    def set_save(self, save: bool): self.save = save
    def set_xlim(self, x_lim: list): self.x_lim = x_lim
    def set_bins(self, bins: int): self.bins = bins
    def set_title(self, title: str): self.title = title
    def set_xlabel(self, x_label: str): self.x_label = x_label
    def set_ylabel(self, y_label: str): self.y_label = y_label
    def set_tail(self, tail: bool): self.tail = tail
    def set_tail_ylim(self, tail: list): self.tail_ylim = tail
    def set_density(self, density: bool): self.density = density
    def set_position_tail(self, postition: list): self.position_tail = postition
    def set_hist_type(self, hist_type: str): self.hist_type = hist_type
    def set_mode(self, mode_: bool): self.mode = mode_
    def set_mean(self, mean_: bool): self.mean = mean_
    def set_outfile(self, outfile: bool): self.outfile = outfile
    def set_y_ticks_tails(self, ticks: int): self.y_ticks_tail = ticks
    

def plot_statistics_words(params: StatisticWordsParameters, df: pd.DataFrame, dimension, plot_sentiment=True):
    """
    dimension: [stamm_] + length|words|sentences
    filter: lambda
    """
    add_extra_info = False
    if type(dimension) != list:
        dimensions = [dimension]
    else:
        dimensions = dimension
        add_extra_info = True
    # fig, axs = plt.subplots(figsize=(5,4))
    fig, axs = plt.subplots()
    legend = []
    mode_colors = list(mcolors.BASE_COLORS)
    mean_colors = list(mcolors.BASE_COLORS)
    nColors = len(mode_colors)
    sub_axes = None
    for kk, dimension in enumerate(dimensions):
        n_bins = int(np.sqrt(len(df[dimension]))) if params.bins is None else params.bins
        hist_params = dict(alpha=0.3, bins=n_bins, ec='k', histtype=params.hist_type, density=params.density)
        axs.hist(df[dimension], **hist_params)
        legend.append(dimension.replace('_', ' '))
        if plot_sentiment:
            axs.hist(df[df['sentiment'] == 1][dimension], **hist_params)
            axs.hist(df[df['sentiment'] == 0][dimension], **hist_params)
            legend += ['negative', 'positive']
        a = df[dimension].to_numpy()
        if params.mode:
            mode = stats.mode(a).mode
            print(" dimension: ", dimension, " mode: ", mode)
            axs.axvline(mode, linestyle='dashed', linewidth=2, color=mode_colors[kk % nColors])
            legend.append('mode' + ('' if not add_extra_info else ' ' + dimension.replace('_', ' ')))
        if params.mean:
            mean_ = a.mean()
            print(" dimension: ", dimension, " mean: ", mean_)
            axs.axvline(mean_, linestyle='dashed', linewidth=2, color=mean_colors[(kk + 5) % nColors])
            legend.append('mean' + ('' if not add_extra_info else ' ' + dimension.replace('_', ' ')))
        if params.x_lim: axs.set_xlim(params.x_lim)
        if params.x_label: axs.set_xlabel(params.x_label)
        if params.y_label: axs.set_ylabel(params.y_label)
        if not params.save: axs.set_title(params.title)
        axs.legend(legend)
        if params.tail: 
            assert params.x_lim is not None
            if sub_axes is None:
                sub_axes = plt.axes(params.position_tail) 
            max_x = df[dimension].max()
            sub_axes.hist(df[dimension], **hist_params)
            sub_axes.xaxis.set_major_locator(plt.MaxNLocator(params.y_ticks_tail))
            if plot_sentiment:
                sub_axes.hist(df[df['sentiment'] == 1][dimension], **hist_params, label='no_legend')
                sub_axes.hist(df[df['sentiment'] == 0][dimension], **hist_params, label='no_legend')
            sub_axes.set_xlim([params.x_lim[1], max_x])
            if params.tail_ylim is not None: sub_axes.set_ylim(params.tail_ylim)
            sub_axes.grid()
    axs.grid()
    if params.save:
        print(" saving file to {}".format(params.outfile))
        fig.savefig(params.outfile, bbox_inches='tight')
    else:
        plt.show()

def plot_emoticons_heatmap(params: StatisticWordsParameters, df: pd.DataFrame):
    # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # df['positive_emoticons'] = df.reviews.apply(lambda x : len(re.findall('positive_emoticon', x)))
    # df['negative_emoticons'] = df.reviews.apply(lambda x : len(re.findall('negative_emoticon', x)))
    import seaborn as sn
    array = [
        [
            len(df[(df.negative_emoticons > 0) & (df.sentiment == 0)]), 
            len(df[(df.negative_emoticons > 0) & (df.sentiment == 1)]), 
            ],
        [
            len(df[(df.positive_emoticons > 0) & (df.sentiment == 0)]),
            len(df[(df.positive_emoticons > 0) & (df.sentiment == 1)]),
            ]
    ]
    df_cm = pd.DataFrame(array, ['negative emoticons', 'positive emoticons'], ['negative reviews', 'positive reviews'])
    sn.set(font_scale=1.2) # for label size
    sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16}, vmin=0, vmax=140) # font size
    if params.save:
        print(" saving file to {}".format(params.outfile))
        plt.savefig(params.outfile, bbox_inches='tight')
    else:
        plt.show()

def plot_venn_diagramm_for_words(params: StatisticWordsParameters, directory):
    for prefix in ['', '_stamm']:
        names = [
            'frequencies_words{}_common',
            'frequencies_words{}_only_pos',
            'frequencies_words{}_only_neg',
        ]
        df_w_common = pd.read_csv(os.path.join(directory, names[0].format(prefix) + '.csv'), sep=';', index_col=False)
        df_w_pos    = pd.read_csv(os.path.join(directory, names[1].format(prefix) + '.csv'), sep=';', index_col=False)
        df_w_neg    = pd.read_csv(os.path.join(directory, names[2].format(prefix) + '.csv'), sep=';', index_col=False)
        fig, axes = plt.subplots()
        title = 'vor' if prefix == '' else 'nach'
        axes.set_title("Wörte in Reviews nach Sentiment {} Bereiningung".format(title), fontsize=16)
        negative = len(df_w_neg)
        postive = len(df_w_pos)
        common = len(df_w_common)
        venn2(subsets = [negative, common, postive, ], set_labels=['Negative Reviews', 'Positive Reviews'], ax=axes)
        axes.grid()

        if params.save:
            out_file = params.outfile + prefix + '.pdf'
            print(" saving file to {}".format(out_file))
            fig.savefig(out_file, bbox_inches='tight')
    if params.save:
        pass
    else:
        plt.show()

def prepare_data_for_grid_search(results_df, model_name):
    w2v_path="/googleW2v/word2vec-google-news-300.gz"
    results_df = results_df.sort_values(by=["rank_test_score"])
    if model_name.startswith("RandomForest"):
        results_df = results_df.set_index(results_df["params"].apply(
            lambda x: "_".join(
                str(val).replace(w2v_path, '')
                        .replace('1', '')
                        .replace('5', '') 
                        .replace('reviews_no_stopwords', 'R.noS.') 
                        .replace('reviews_no_punctuation', 'R.noP.') 
                        .replace('stamm_no_punctuation', 'S.noP.') 
                        .replace('stamm_no_stop_punct', 'S.noS.P.') 
                        for val in x.values()
                ))).rename_axis("hyper-parameter")
    else:
        results_df = results_df.set_index(results_df["params"].apply(
            lambda x: "_".join(
                str(val).replace(w2v_path, '')
                        .replace('', '') 
                        for val in x.values()
                    ))).rename_axis("hyper-parameter")
    results_df[["rank_test_score", "mean_test_score", "std_test_score"]]
    return results_df

def plot_scores_grid_search(params: StatisticWordsParameters, results_df, model_name):
    import seaborn as sns
    # results_df.params.iloc[0]
    results_df = prepare_data_for_grid_search(results_df, model_name)
    model_scores = results_df.filter(regex=r"split\d*_test_score")
    fig, ax = plt.subplots()
    sns.lineplot(
        data=model_scores.transpose().iloc[:30],
        dashes=False,
        palette="Set1",
        marker="o",
        alpha=0.5,
        ax=ax,
    )
    ax.set_ylim(params.x_lim)
    ax.set_title("Cross Validation results for {}".format(model_name))
    ax.grid('minor')
    ax.set_xlabel("CV test fold", size=12, labelpad=10)
    ax.set_ylabel("Model accuracy", size=12)
    ax.tick_params(bottom=True, labelbottom=False)
    ax.legend(ncols=2)
    if params.save:
        print(" saving file to {}".format(params.outfile))
        fig.savefig(params.outfile, bbox_inches='tight')
    else:
        plt.show()

def plot_correlation_heatmap(params: StatisticWordsParameters, results_df, model_name):
    results_df = prepare_data_for_grid_search(results_df, model_name)
    model_scores = results_df.filter(regex=r"split\d*_test_score")
    corr = model_scores.transpose().corr()
    import seaborn as sns
    fig, ax = plt.subplots()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values, vmin=-1, vmax=1, annot=True)
    if params.save:
        print(" saving file to {}".format(params.outfile))
        fig.savefig(params.outfile, bbox_inches='tight')
    else:
        plt.show()