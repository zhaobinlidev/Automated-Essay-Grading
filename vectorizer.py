import sklearn.feature_extraction
import scipy
import numpy as np
import time
import zipfile


class Vectorizer:
    def __init__(self):
        self.tfidfVectorizer = sklearn.feature_extraction.text.TfidfVectorizer
        self.countVectorizer = sklearn.feature_extraction.text.CountVectorizer
        self.svd = scipy.sparse.linalg.svds

    def run(self, essay_train, essay_test, dimensions=500, SVD_write_path=None, SVD=None, ngram_range=None,
            verbose=False):
        '''

        :param essay_train: train essays
        :param essay_test: test essays
        :param dimensions: dimension to reduce to
        :param SVD_write_path: (optional) where to write the SVD components to
        :param SVD: (optional) use precalculated SVD components
        :param ngram_range: (optional) what ngram range to use
        :param verbose: print error messages
        :return: reduced train and test ngram-by-essay matrix, the ngrams and the SVD components
        '''
        train_X, tokens = self.vectorize(essay_train, verbose, self.tfidfVectorizer, ngram_range=ngram_range)

        if SVD is None:
            U, S, VT = self.get_svd_components(train_X, verbose, SVD_write_path)
        else:
            U, S, VT = SVD

        train_X_r = self.get_reduced_matrix(train_X, U, S, VT, dimensions)

        test_X_r = None

        if essay_test is not None:
            test_X, _ = self.vectorize(essay_test, verbose, self.tfidfVectorizer, vocabulary=tokens,
                                       ngram_range=ngram_range)

            test_X_r = self.get_reduced_matrix(test_X, U, S, VT, dimensions)

            if test_X_r is None: #singular matrix
                return None

        return train_X_r, test_X_r, tokens, (U, S, VT)

    def vectorize(self, essay_set, verbose, vectorize_fn, **kwargs):
        '''
        :param essay_set:
        :param args:
        :return: n*m matrix
        '''

        start_time = time.time()

        vector_class = vectorize_fn(**kwargs)
        vector_X = vector_class.fit_transform(essay_set)

        elapsed_time = time.time() - start_time
        tokens = vector_class.get_feature_names()

        if verbose:
            print("Start time to vectorize: " + str(time.localtime(start_time)))
            print("Elapsed time to vectorize: " + str(elapsed_time / 60))
            print()
            print("Token No.: " + str(len(tokens)))

        return vector_X.T.toarray().astype("float"), tokens

    def get_reduced_matrix(self, n_by_m_matrix, U, S, VT, dimensions):
        '''

        :param n_by_m_matrix:
        :param U:
        :param S:
        :param VT:
        :param dimensions:
        :return: m*k matrix
        '''

        U = U[:, :dimensions]
        S = S[:dimensions, :dimensions]
        VT = VT[:dimensions, :]

        try:
            S_inverse = np.linalg.inv(S)
        except np.linalg.linalg.LinAlgError: #singular matrix
            return None

        return n_by_m_matrix.T @ (U @ np.linalg.inv(S_inverse))

    def get_svd_components(self, matrix, verbose, write_path=None):
        '''

        :param matrix:
        :param dimensions:
        :return: Singular values U or n*d
                 Eigenvalues S or d*d
                 Eigenvalues VT or d*m
        '''
        start_time = time.time()
        U, s, VT = self.svd(matrix, k=min(matrix.shape) - 1)
        S = np.diag(s)

        elapsed_time = time.time() - start_time

        if verbose:
            print("Start time to SVD: " + str(time.localtime(start_time)))
            print("Elapsed time to SVD: " + str(elapsed_time / 60))
            print()

        if write_path is not None:
            np.savez_compressed(write_path, U=U, S=S, VT=VT, time=np.array(elapsed_time / 60))

        return U, S, VT

    def read_svd_components(self, SVD_path):
        try:
            SVD = np.load(SVD_path)
            return SVD["U"], SVD["S"], SVD["VT"]
        except zipfile.BadZipfile:
            return None

    def read_dataset(self, dataset_path):
        dataset = np.load(dataset_path)
        return dataset["essay_train_set"], dataset["essay_valid_set"], dataset["essay_test_set"], \
               dataset["score_train_set"], dataset["score_valid_set"], dataset["score_test_set"]
