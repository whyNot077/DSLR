import numpy as np
import math
from lib.ft_statistics import ft_min, ft_max, ft_mean


class LogisticRegression:
    def __init__(self, solver='gd', max_iter=100, verbose=0, batch_size=64):
        self.max_iter_ = max_iter
        self.verbose_ = verbose
        if solver == 'sgd':
            self.batch_size_ = 1
        elif solver == 'mbgd':
            self.batch_size_ = batch_size
        else:
            self.batch_size_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_classes = ft_max(y, skipna=False) + 1
        self.coef_ = np.zeros((n_classes, X.shape[1]))
        self.intercept_ = np.zeros(n_classes)
        self.fit_history_ = [self.__get_cost(X, y)]
        while len(self.fit_history_) - 1 < self.max_iter_:
            self.__fit_one_epoch(X, y)
        return self

    def predict(self, X: np.ndarray):
        y_pred = np.array([self.__predict_sample(sample) for sample in X],
                          dtype='object')
        return y_pred

    def get_params(self):
        return self.coef_, self.intercept_

    def set_params(self, params):
        self.coef_, self.intercept_ = params
        return self

    def get_fit_history(self):
        return self.fit_history_

    def __fit_one_epoch(self, X, y):
        if self.batch_size_ is not None:
            X, y = LogisticRegression.__shuffle_data(X, y)
        if self.batch_size_ is None:
            self.__train(X, y)
            return
        for batch_num in range((len(X) - 1) // self.batch_size_ + 1):
            start_idx = batch_num * self.batch_size_
            end_idx = ft_min(start_idx + self.batch_size_, len(X))
            self.__train(X[start_idx:end_idx], y[start_idx:end_idx])
            if len(self.fit_history_) - 1 == self.max_iter_:
                break

    def __train(self, X, y, label=None):
        if label is None:
            for label in range(len(self.coef_)):
                self.__train(X, y, label)
            cost = self.__get_cost(X, y)
            iteration = len(self.fit_history_)
            if iteration % 10 == 0:
                print(f'iteration = {iteration}, cost = {cost}')
            self.fit_history_.append(cost)
            return
        y = np.where(y == label, 1, 0)
        h = np.array([self.__estimate_proba(sample, label) for sample in X])
        a = h - y
        tmp_coef = np.array([ft_mean(a * X[:, j], False)
                             for j in range(X.shape[1])])
        tmp_intercept = ft_mean(a, False)
        self.coef_[label] -= tmp_coef
        self.intercept_[label] -= tmp_intercept

    def __get_cost(self, X, y, label=None):
        if label is None:
            return sum(self.__get_cost(X, y, label)
                       for label in range(len(self.coef_)))
        y = np.where(y == label, 1, 0)
        h = np.array([self.__estimate_proba(sample, label) for sample in X])
        J = -ft_mean(y * np.log(h) + (1 - y) * np.log(1 - h), False)
        return J

    def __predict_sample(self, sample):
        proba = [self.__estimate_proba(sample, label)
                 for label in range(len(self.coef_))]
        label = proba.index(ft_max(proba, skipna=False))
        return label

    def __estimate_proba(self, sample, label):
        z = sum(self.coef_[label] * sample) + self.intercept_[label]
        return LogisticRegression.__sigmoid(z)

    def __shuffle_data(X, y):
        rand = np.random.permutation(len(X))
        return X[rand], y[rand]

    def __sigmoid(z):
        return 1 / (1 + math.exp(-z))
