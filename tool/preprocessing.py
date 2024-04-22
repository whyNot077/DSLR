import pandas as pd
from statistics import ft_min, ft_max


class MinMaxScaler:
    def fit(self, X: pd.DataFrame):
        self.data_min_ = X.agg(ft_min)
        self.data_range_ = X.agg(ft_max) - self.data_min_
        return self

    def transform(self, X: pd.DataFrame):
        Xt = X.apply(lambda row: (row - self.data_min_) / self.data_range_,
                     axis=1).to_numpy()
        return Xt
