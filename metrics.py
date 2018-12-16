import sklearn.metrics
import pandas as pd
import numpy as np


class Metrics:
    def __init__(self):
        self.cosine = sklearn.metrics.pairwise.cosine_similarity
        self.confusion_matrix = sklearn.metrics.confusion_matrix
        self.f1_score = sklearn.metrics.f1_score

    def run(self, train_Z, test_Z, scores_train, scores_test, results_write_path=None, verbose=False):
        '''
        Assign scores to the test set giving the training set matrix
        and scores and calculate the accurary
        :param train_Z: the reduced train ngram-by-essay matrix
        :param test_Z: the reduced test ngram-by-essay matrix
        :param scores_train: the train essay scores
        :param scores_test: the test essay scores
        :param results_write_path: path to write the results to
        :param verbose: print statistics
        :return: return statistics
        '''
        cosine_matrix = self.get_cosine_matrix(train_Z, test_Z)
        max_sim_scores = self.get_max_similarity(cosine_matrix, scores_train)

        score_F1 = self.get_F1_score(max_sim_scores, scores_test)
        max_sim_acc = self.get_accuracy_within_1(max_sim_scores, scores_test)
        max_sim_err = self.get_absolute_error(max_sim_scores, scores_test)

        if verbose:
            print("F1 score: " + str(score_F1))
            print("Accuracy within 1 score : " + str(max_sim_acc))
            print("Average error : " + str(max_sim_err))
            print()

        statistics = {"max_sim_scores": max_sim_scores, "score_F1": score_F1, "max_sim_acc": max_sim_acc,
                "max_sim_err": max_sim_err}

        if results_write_path is not None:
            statistics_df = pd.DataFrame(statistics)
            statistics_df.to_csv(results_write_path + ".csv.gz", index=False, compression='gzip')

        return statistics

    def get_cosine_matrix(self, train_Z, test_Z):
        '''
        :param train_Z: j
        :param test_Z: i
        :return: i*j
        '''
        return self.cosine(test_Z, train_Z)

    def get_max_similarity(self, cosine_matrix, scores_train):
        max_sim_idx = np.argmax(cosine_matrix, axis=1)
        scores_test = np.array([scores_train[i] for i in max_sim_idx])
        return scores_test

    def get_accuracy_within_1(self, predicted_scores, scores):
        return sum(abs(predicted_scores - scores) <= 1) / len(predicted_scores)

    def get_absolute_error(self, predicted_scores, scores):
        return sum(abs(predicted_scores - scores)) / len(predicted_scores)

    def get_confusion_matrix(self, predicted_scores, scores):
        return self.confusion_matrix(scores, predicted_scores, labels=[i for i in range(1,11)])

    def get_F1_score(self, predicted_scores, scores):
        return self.f1_score(scores, predicted_scores, labels=[i for i in range(1,11)], average='weighted')
