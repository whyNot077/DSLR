import numpy as np
import math
from statistics import ft_min, ft_max, ft_mean


class LogisticRegression:
    def __init__(self, solver='gd', max_iter=100, verbose=0, batch_size=32):
        self.max_iter_ = max_iter
        self.verbose_ = verbose
        if solver == 'sgd':
            self.batch_size_ = 1
        elif solver == 'mbgd':
            self.batch_size_ = batch_size
        else:
            self.batch_size_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.coef_ = np.zeros((n_classes, X.shape[1]))
        self.intercept_ = np.zeros(n_classes)
        for epoch in range(1, self.max_iter_ + 1):
            self.__fit_one_epoch(X, y)
            if epoch % 10 == 0:
                cost = sum(self.__get_cost(X, y, label)
                           for label in range(n_classes))
                print(f'epoch = {epoch}, cost = {cost}')
        return self

    def predict(self, X: np.ndarray):
        y_pred = np.array([self.__predict_sample(sample) for sample in X],
                          dtype='object')
        return y_pred

    def __fit_one_epoch(self, X, y):
        if self.batch_size_ is not None:
            X, y = LogisticRegression.__shuffle_data(X, y)
        for label in range(len(self.classes_)):
            if self.batch_size_ is None:
                self.__train(X, y, label)
                continue
            for batch_num in range((len(X) - 1) // self.batch_size_ + 1):
                start_idx = batch_num * self.batch_size_
                end_idx = ft_min(start_idx + self.batch_size_, len(X))
                self.__train(X[start_idx:end_idx], y[start_idx:end_idx], label)

    def __train(self, X, y, label):
        y = np.where(y == self.classes_[label], 1, 0)
        h = np.array([self.__estimate_proba(sample, label) for sample in X])
        a = h - y
        tmp_coef = np.array([ft_mean(a * X[:, j]) for j in range(X.shape[1])])
        tmp_intercept = ft_mean(a)
        self.coef_[label] -= tmp_coef
        self.intercept_[label] -= tmp_intercept

    def __get_cost(self, X, y, label=None):
        if label is None:
            return sum(self.__get_cost(X, y, label)
                       for label in range(len(self.classes_)))
        y = np.where(y == self.classes_[label], 1, 0)
        h = np.array([self.__estimate_proba(sample, label) for sample in X])
        J = -ft_mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        return J

    def __predict_sample(self, sample):
        n_classes = len(self.classes_)
        proba = [self.__estimate_proba(sample, label)
                 for label in range(n_classes)]
        label = proba.index(ft_max(proba))
        return self.classes_[label]

    def __estimate_proba(self, sample, label):
        z = sum(self.coef_[label] * sample) + self.intercept_[label]
        return LogisticRegression.__sigmoid(z)

    def __shuffle_data(X, y):
        rand = np.random.permutation(len(X))
        return X[rand], y[rand]

    def __sigmoid(z):
        return 1 / (1 + math.exp(-z))
