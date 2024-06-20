from sys import argv
from lib.datasets import load_magic_hat
from lib.ft_statistics import ft_mean
from lib.preprocessing import MinMaxScaler
from lib.linear_model import LogisticRegression
import pickle


def main():
    if len(argv) != 2:
        print(f'[Usage] python3 {argv[0]} dataset_train.csv')
        return
    try:
        X_train, y_train = load_magic_hat(argv[1])
        y_train = y_train['Hogwarts House'].to_numpy()
        column_mean = [ft_mean(X_train[column].dropna())
                       for column in X_train.columns]
        X_train = X_train.fillna({X_train.columns[i]: column_mean[i]
                                  for i in range(len(column_mean))})
        scaler = MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        lr = LogisticRegression(verbose=1)
        lr.fit(X_train, y_train)
        with open('weights.p', 'wb') as f:
            pickle.dump((column_mean, scaler.get_params(),
                         lr.get_params()), f)
    except (Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == '__main__':
    main()
