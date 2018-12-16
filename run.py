#!/usr/bin/env python
# coding: utf-8

import vectorizer
import metrics
import numpy as np
import pandas as pd
import math
import glob
import ast
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


class Optimizer:
    def __init__(self):
        self.Vectorizer = vectorizer.Vectorizer()
        self.Metrics = metrics.Metrics()
        self.ngram_list = self.get_ngrams()

    def run(self, dataset_path, SVD_write_dir, results_write_dir, SVD_read_dir=None, question_to_optimize=None,
            ngram_range_to_optimize=None):
        '''
        Run “run.main()” to get the optimal ngram-dimension pair
        and the corresponding F1 scores in the validation and
        test set across all 8 questions.
        a.	For every question, iterates through 18 ngram ranges
        to build ngram-by-documents to apply SVD to (using vectorizer).
        b.	Truncate the SVD components in 30 dimensions and use them
        to reduce the training and validation set. Assign the scores
        and calculate the F1 scores on the validation set. For every
        ngram range the optimal dimension is the one which gives the
        greatest F1 score on the validation set (using metrics).
        c.	Then try the 18 ngram with optimal dimensions on the test
        set to get new F1 scores. The ngram-dimension pair which gives
        the greatest F1 score on the test set is the optimal ngram and
        dimension to use in a question.
        d.	Iterate across all 8 questions and store the validation and
        test statistics in a gzipped csv table in the “data” Folder

        :param dataset_path: path containing the essay and scores data
        :param SVD_write_dir: directory to store the SVD components
        :param results_write_dir: directory to write the results to
        :param SVD_read_dir: (optional) directory to read the SVD components
         when using precalculated SVD components
        :param question_to_optimize: (optional) select which question 1-8
        to optimize, or else optimize all questions
        :param ngram_range_to_optimize: (optional) select which ngram ranges
        to optimize, or else optimize all ngram ranges by calling get_ngrams()
        :return: None. Store the validation and test results in the results_write_dir
        '''
        essay_train_set, essay_valid_set, essay_test_set, score_train_set, score_valid_set, score_test_set = \
            self.read_dataset(dataset_path)

        SVD_by_question_by_ngram = [[[] for ngram in range(len(self.ngram_list))] for question_no in range(9)]
        if SVD_read_dir is not None:
            for SVD_path in glob.iglob(SVD_read_dir + '/**/*SVD*', recursive=True):
                _, question_no, ngram_range = SVD_path.split('.')[0].split("_")
                question_no = int(question_no)
                ngram_range = ast.literal_eval(ngram_range)
                SVD = self.Vectorizer.read_svd_components(SVD_path)
                if SVD is not None:
                    SVD_by_question_by_ngram[question_no][self.ngram_list.index(ngram_range)] \
                        .append(SVD)

        if ngram_range_to_optimize is None:
            ngram_ranges = self.ngram_list
        else:
            ngram_ranges = ngram_range_to_optimize

        for ngram_range in ngram_ranges:

            metrics_valid_list = []
            metrics_test_list = []

            if question_to_optimize is None:
                question_list = [i for i in range(8)]
            else:
                question_list = [i - 1 for i in question_to_optimize]

            for question_no in question_list:

                print({"question_no": question_no + 1, "ngram_range": ngram_range})
                if len(SVD_by_question_by_ngram[question_no][self.ngram_list.index(ngram_range)]) == 0:
                    SVD = None
                else:
                    SVD = SVD_by_question_by_ngram[question_no][self.ngram_list.index(ngram_range)]

                train_X, _, tokens, SVD = \
                    self.Vectorizer.run(essay_train_set[question_no], essay_test=None,
                                        dimensions=len(essay_train_set[question_no]),
                                        # no dimensionality reduction
                                        SVD_write_path=SVD_write_dir + "/SVD_" + str(question_no) + "_" + str(
                                            ngram_range),
                                        SVD=SVD,
                                        ngram_range=ngram_range)

                optimum_dimension = 0
                max_score_F1_valid = 0

                for dimensions in self.get_dimensions(len(essay_train_set[question_no])):
                    if dimensions < 2:
                        continue

                    args = \
                        self.Vectorizer.run(essay_train_set[question_no], essay_valid_set[question_no],
                                            dimensions=dimensions,
                                            SVD=SVD,
                                            ngram_range=ngram_range)

                    if args is not None:
                        train_X_r, valid_X_r, _, _ = args
                    else: #singular matrix
                        continue

                    statistics_valid = self.Metrics.run(train_X_r, valid_X_r, score_train_set[question_no],
                                                        score_valid_set[question_no])
                    statistics_valid["max_sim_scores"] = str(statistics_valid["max_sim_scores"].tolist())

                    metrics_valid_dict = dict({"question_no": question_no + 1, "ngram_range": str(ngram_range),
                                               "dimensions": dimensions}, **statistics_valid)
                    metrics_valid_list.append(pd.DataFrame(metrics_valid_dict, index=[0]))

                    if statistics_valid["score_F1"] > max_score_F1_valid:
                        max_score_F1_valid = statistics_valid["score_F1"]
                        optimum_dimension = dimensions

                train_X_r, test_X_r, _, _, = \
                    self.Vectorizer.run(essay_train_set[question_no], essay_test_set[question_no],
                                        dimensions=optimum_dimension,
                                        SVD=SVD,
                                        ngram_range=ngram_range)

                statistics_test = \
                    self.Metrics.run(train_X_r, test_X_r, score_train_set[question_no], score_test_set[question_no])
                statistics_test["max_sim_scores"] = str(statistics_test["max_sim_scores"].tolist())

                metrics_test_dict = dict({"question_no": question_no + 1, "optimum_ngram_range": str(ngram_range),
                                          "optimum_dimensions": optimum_dimension}, **statistics_test)
                metrics_test_list.append(pd.DataFrame(metrics_test_dict, index=[0]))

                print(
                    {"score_F1_test": statistics_test["score_F1"], "max_acc_test": statistics_test["max_sim_acc"],
                     "optimum_dimensions": optimum_dimension})
                print()

            metrics_valid_df = pd.concat([df for df in metrics_valid_list], ignore_index=True)
            metrics_valid_df.to_csv(results_write_dir + "/metrics_valid_" + str(ngram_range) + "_" + '.csv.gz',
                                    index=False, compression='gzip')

            metrics_test_df = pd.concat([df for df in metrics_test_list], ignore_index=True)
            metrics_test_df.to_csv(results_write_dir + "/metrics_test_" + str(ngram_range) + "_" + '.csv.gz',
                                   index=False,
                                   compression='gzip')

    def read_dataset(self, dataset_path):
        dataset = np.load(dataset_path)
        return dataset["essay_train_set"], dataset["essay_valid_set"], dataset["essay_test_set"], \
               dataset["score_train_set"], dataset["score_valid_set"], dataset["score_test_set"]

    def read_svd_components(self, SVD_path):
        SVD = np.load(SVD_path)
        return SVD["U"], SVD["S"], SVD["VT"]

    def get_ngrams(self):
        ngram_ranges = []
        for start_range in (1, 3, 5, 10):
            for end_range in (1, 3, 5, 10, 15, 20):
                if start_range <= end_range:
                    ngram_ranges.append(tuple([start_range, end_range]))
        return sorted(ngram_ranges, key=lambda ngram_range: ngram_range[1] - ngram_range[0])

    def get_dimensions(self, max_dimensions):
        return np.linspace(2, max_dimensions, num=30, dtype=int)


def main():
    optimizer = Optimizer()
    optimizer.run(dataset_path="data/dataset.npz", SVD_write_dir="data", results_write_dir="data",
                  question_to_optimize=None, ngram_range_to_optimize=None)

if __name__ == '__main__':
    main()
