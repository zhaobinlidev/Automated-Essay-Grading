# coding=utf-8
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast

plt.style.use('bmh')
params = {
    'legend.fontsize': 12 * 1.5,
    'xtick.labelsize': 15 * 1.5,
    'ytick.labelsize': 15 * 1.5,
    'axes.labelsize': 20 * 1.5,
    'axes.titlesize': 20 * 1.5,
    'figure.figsize': (20, 10),
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'legend.borderaxespad': 0.,
    'legend.loc': 'upper left'
}
mpl.rcParams.update(params)


class Visualizer:
    def __init__(self):
        pass

    def run(self, results_dir):
        '''
        Run “visualizer.main()” to get all the graphs in the “img” Folder
        :param results_dir: directory containing the results
        :return: generate matplotlib graphs in "img" Folder and return the optimum_results_df
        '''
        metrics_valid_df, metrics_test_df = self.get_results(results_dir)
        optimum_results_df = self.get_optimum_results(metrics_test_df)
        ngram_keys = self.get_ngrams_to_plt(metrics_test_df, optimum_results_df)

        self.plt_optimum_dimensions_by_question(optimum_results_df)
        self.plt_F1_score_by_ngram(metrics_test_df)
        self.plt_optimum_dimension_by_ngram(metrics_test_df)
        self.plt_optimum_F1_score_by_question(optimum_results_df)
        self.plt_F1_score_against_dimension(metrics_valid_df)
        self.plt_ngram_by_dimension_per_question(metrics_valid_df)

        for ngram_key in ngram_keys:
            self.plt_ngram_by_question(metrics_test_df, ngram_key)

        self.plt_dimension_by_ngram(metrics_test_df)

        return optimum_results_df

    def get_results(self, results_dir):
        metrics_valid_list = []
        metrics_test_list = []

        for path in glob.iglob(results_dir + '/**/*.csv*', recursive=True):
            if 'metrics' in path:
                if 'valid' in path:
                    metrics_valid_list.append(pd.read_csv(path))
                elif 'test' in path:
                    metrics_test_list.append(pd.read_csv(path))

        metrics_valid_df = pd.concat([df for df in metrics_valid_list], ignore_index=True)
        metrics_test_df = pd.concat([df for df in metrics_test_list], ignore_index=True)

        return metrics_valid_df, metrics_test_df

    def get_optimum_results(self, metrics_test_df):
        idx = metrics_test_df.groupby(['question_no'])['score_F1'] \
                  .transform(max) == metrics_test_df['score_F1']

        optimum_results_df = metrics_test_df[idx].sort_values(['question_no']).loc[:,
                             metrics_test_df.columns != 'max_sim_scores']

        with open('optimized_results.tex', 'w') as f:
            optimized_latex = optimum_results_df.rename(
                {"question_no": 'Question', "optimum_ngram_range": "ngram", "optimum_dimensions": "Dims",
                 "score_F1": "F1", "max_sim_acc": "% correct", "max_sim_err": "Absolute Error"}, axis=1). \
                round(3).to_latex(index=False, bold_rows=True)
            f.write(optimized_latex)

        return optimum_results_df

    def get_ngrams_to_plt(self, metrics_test_df, optimum_results_df):
        ngram_keys = [[ast.literal_eval(ngram) for ngram in optimum_results_df["optimum_ngram_range"].unique()]]

        ngram_ranges = [ast.literal_eval(ngram) for ngram in metrics_test_df["optimum_ngram_range"].unique()]
        ngram_keys.append([ngram for ngram in ngram_ranges if ngram[1] - ngram[0] == 0])
        for start in (1, 3, 5):
            ngram_keys.append([ngram for ngram in ngram_ranges if ngram[0] == start])

        ngram_keys_sorted = []
        for ngram_key in ngram_keys:
            ngram_key.sort(key=lambda ngram: (ngram[1] - ngram[0], ngram[0]))
            ngram_keys_sorted.append([str(ngram) for ngram in ngram_key])

        return ngram_keys_sorted

    def plt_optimum_dimensions_by_question(self, optimum_results_df):
        ax = optimum_results_df.plot(x="question_no", y="optimum_dimensions", kind='bar', legend=False)
        ax.set_xlabel('Question no')
        ax.set_ylabel("Optimum Dimensions")
        ax.set_title('Optimum Dimensions by Questions')
        plt.savefig(
            fname='img/optimum_dimensions_by_question.png')

    def plt_F1_score_by_ngram(self, metrics_test_df):
        question = 1
        metrics_test_plt = metrics_test_df.loc[(metrics_test_df['question_no'] == question)].sort_values(by="score_F1",
                                                                                                  ascending=False)
        ax = metrics_test_plt.plot(x="optimum_ngram_range", y="score_F1", kind='bar', legend=False)
        ax.set_xticklabels([ngram for ngram in metrics_test_plt["optimum_ngram_range"].values])
        ax.set_xlabel('n-gram Range')
        ax.set_ylabel("F1 score")
        ax.set_title('Question ' + str(question) + ': F1 score by n-gram Range')
        plt.savefig(
            fname='img/F1_score_by_ngram_qn_' + str(question) + '.png')

    def plt_optimum_dimension_by_ngram(self, metrics_test_df):
        question = 1
        metrics_test_plt = metrics_test_df.loc[(metrics_test_df['question_no'] == question)].sort_values(
            by="optimum_dimensions",
            ascending=False)
        ax = metrics_test_plt.plot(x="optimum_ngram_range", y="optimum_dimensions", kind='bar', legend=False)
        ax.set_xticklabels([ngram for ngram in metrics_test_plt["optimum_ngram_range"].values])
        ax.set_xlabel('n-gram Range')
        ax.set_ylabel("Optimal Dimension")
        ax.set_title('Question ' + str(question) + ': Optimal Dimension by n-gram Range')
        plt.savefig(
            fname='img/optimum_dimensions_by_ngram_qn_' + str(question) + '.png')


    def plt_optimum_F1_score_by_question(self, optimum_results_df):
        ax = optimum_results_df.plot(x="question_no", y="score_F1", legend=False, kind='bar')
        ax.set_xticklabels(
            [str(question) for question in optimum_results_df["question_no"].values])
        ax.set_xlabel('Question no')
        ax.set_ylabel("F1 score")
        ax.set_title('F1 score by Questions')
        plt.savefig(
            fname='img/optimum_F1_score_by_question.png')


    def plt_F1_score_against_dimension(self, metrics_valid_df):
        ngram = '(5, 15)'
        question = 1
        metrics_valid_plt = metrics_valid_df.loc[
            (metrics_valid_df['question_no'] == question) & (metrics_valid_df['ngram_range'] == ngram)]

        ax = metrics_valid_plt.plot(x="dimensions", y='score_F1', legend=False)
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('F1 score')
        ax.set_title(
            'Question ' + str(question) + ', n-gram ' + ngram + ': ' + 'F1 score by dimensions')
        plt.savefig(
            fname='img/F1_score_against_dimension_score_F1_question_ ' + str(question) + '.png')

    def plt_ngram_by_dimension_per_question(self, metrics_valid_df):

        fig = plt.figure(figsize=(40, 30))
        fig.suptitle('F1 score by n-grams', fontsize=40)
        for question in range(1, 9):

            axes = fig.add_subplot(4, 2, question)
            metrics_valid_per_question = metrics_valid_df[metrics_valid_df['question_no'] == question]
            metrics_valid_plt = metrics_valid_per_question.groupby(['ngram_range', 'dimensions'])['score_F1'].mean()

            ax = metrics_valid_plt.unstack(level=0).plot(ax=axes, kind='line', sharey=True, sharex=True)
            ax.set_xlabel('Dimensions')
            ax.set_ylabel('F1 score')
            ax.set_title('Question ' + str(question))
            ax.legend(title='n-grams', bbox_to_anchor=(1.05, 1))

            # q1: save individual images
            if question == 1:
                ax_per_question = metrics_valid_plt.unstack(level=0).plot(kind='line')
                ax_per_question.set_xlabel('Dimensions')
                ax_per_question.set_ylabel('F1 score')
                ax_per_question.set_title('Question ' + str(question) + ': ' + 'F1 score by n-grams')
                ax_per_question.legend(title='n-grams', bbox_to_anchor=(1.05, 1))

                plt.savefig(
                    fname='img/ngram_by_dimension_per_question_score_F1_question_' + str(question) + '.png')

        fig.savefig(
            fname='img/ngram_by_dimension_per_question_score_F1.png')

    def plt_ngram_by_question(self, metrics_test_df, ngram_key):

        metrics_test_ngram = metrics_test_df.loc[metrics_test_df['optimum_ngram_range'].isin(ngram_key)]
        metrics_test_plt = metrics_test_ngram.groupby(['optimum_ngram_range', 'question_no'])['score_F1'].mean()

        ax = metrics_test_plt.unstack(level=0).reindex(ngram_key, axis=1).plot(kind='bar')

        ax.set_xlabel('Question No')
        ax.set_ylabel('F1 score')
        ax.set_title(
            'F1 score by optimum n-gram and question')
        ax.legend(title='Optimum ngram', bbox_to_anchor=(1.05, 1))
        plt.savefig(fname=
                    'img/ngram_by_question_score_F1' + str(ngram_key) + '.png')


    def plt_dimension_by_ngram(self, metrics_test_df):

        metrics_test_plt = metrics_test_df.groupby(['optimum_dimensions', 'optimum_ngram_range'])['score_F1'].mean()
        ax = metrics_test_plt.unstack(level=0).plot(kind='bar')

        ax.set_xlabel('n-gram')
        ax.set_ylabel('F1 score')
        ax.set_title('F1 score by optimum dimension and n-gram')
        legend = ax.legend(labels='', title='Dimensions in ascending order', bbox_to_anchor=(1.05, 1))
        plt.setp(legend.get_title(), fontsize=12 * 1.5)
        plt.savefig(
            fname='img/dimension_by_ngram_score_F1.png')


def main():
    visualizer = Visualizer()
    visualizer.run("data")


if __name__ == '__main__':
    main()
