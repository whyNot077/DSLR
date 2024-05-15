import sys
import pickle
from datasets import load_magic_hat
from statistics import ft_mean
from preprocessing import MinMaxScaler
from linear_model import LogisticRegression
import matplotlib.pyplot as plt


def show_train_history(train_history):
    plt.figure('train history')
    plt.plot(train_history)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()


def main():
    if len(sys.argv) != 2:
        print('[Usage] python3', sys.argv[0], 'dataset_train.csv')
        return
    try:
        X_train, y_train = load_magic_hat(sys.argv[1])
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
        show_train_history(lr.get_fit_history())
    except (Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == '__main__':
    main()
