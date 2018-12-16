# coding=utf-8
import nltk
import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing

class Preprocessor:
    def __init__(self):
        self.essay_pd = None
        self.wordnet_lemmatizer = nltk.WordNetLemmatizer()
        self.english_stopwords = set(nltk.corpus.stopwords.words('english'))
        self.regexp_tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.train_test_split = sklearn.model_selection.train_test_split
        self.scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(1, 10))

    def run(self, essay_set_pathname, write_path=None):
        '''
        Run “preprocessor.main()” to process the essays across all 8 questions:
        a.	Removes stopwords, lemmatizes and tokenizes essays (using clean_and_tokenize)
        b.	Scales the scores to be consistent across all questions to 1 and 10 and
        splitting them into train, validation and test set and then saving the 6 datasets
        - essay_train_set, essay_valid_set, essay_test_set, score_train_set, score_valid_set, score_test_set
        – into dataset.npz in the “data” Folder(using run)
        :param essay_set_pathname: path to essay data
        :param write_path: path to write the parsed essay data
        :param question_to_optimize: essay questions to parse
        :return: train, validation, and test essays and scores
        '''

        self.essay_pd = pd.read_csv(essay_set_pathname, header=0, index_col=0,
                                    usecols=["essay_id", "essay_set", "essay", "rater1_domain1", "rater2_domain1",
                                             "domain1_score"])
        self.essay_pd["tokens"] = self.essay_pd["essay"].apply(self.clean_and_tokenize)
        essay_train_set = []
        essay_valid_set = []
        essay_test_set = []
        score_train_set = []
        score_valid_set = []
        score_test_set = []

        for qn in range(1, 9):
            essay_set = self.essay_pd[self.essay_pd["essay_set"] == qn]
            essay_set_raw_scores = essay_set["domain1_score"].values
            essay_set_scaled_scores = self.scaler.fit_transform(essay_set_raw_scores.reshape(-1, 1))
            essay_set["scaled_scored"] = essay_set_scaled_scores.flatten().astype(int).tolist()
            train_set, not_train_set = sklearn.model_selection.train_test_split(essay_set, test_size=0.4)
            valid_set, test_set = sklearn.model_selection.train_test_split(not_train_set, test_size=0.4)
            essay_train_set.append(train_set["tokens"].values)
            essay_valid_set.append(valid_set["tokens"].values)
            essay_test_set.append(test_set["tokens"].values)
            score_train_set.append(train_set["scaled_scored"].values)
            score_valid_set.append(valid_set["scaled_scored"].values)
            score_test_set.append(test_set["scaled_scored"].values)
        if write_path is not None:
            np.savez(write_path, essay_train_set=np.asarray(essay_train_set),
                     essay_valid_set=np.asarray(essay_valid_set),
                     essay_test_set=np.asarray(essay_test_set),
                     score_train_set=np.asarray(score_train_set),
                     score_valid_set=np.asarray(score_valid_set),
                     score_test_set=np.asarray(score_test_set))
        return essay_train_set, essay_valid_set, essay_test_set, score_train_set, score_valid_set, score_test_set

    def clean_and_tokenize(self, essay):
        tokens = essay.lower().split()
        tokens = [s for s in tokens if "@" not in s]  # remove @substituted words
        tokens = self.regexp_tokenizer.tokenize(" ".join(tokens))  # split np.string_ into words (tokens)
        tokens = [self.wordnet_lemmatizer.lemmatize(t) for t in tokens]  # put words into base form
        tokens = [t for t in tokens if t not in self.english_stopwords]  # remove stopwords
        tokens = [t for t in tokens if len(t) > 2]  # remove short words

        return " ".join(tokens)

    def read_dataset(self, dataset_path):
        dataset = np.load(dataset_path)
        return dataset["essay_train_set"], dataset["essay_valid_set"], dataset["essay_test_set"], \
               dataset["score_train_set"], dataset["score_valid_set"], dataset["score_test_set"]


def main():
    preprocessor = Preprocessor()
    preprocessor.run("training_set.csv", "data/dataset")


if __name__ == '__main__':
    main()
