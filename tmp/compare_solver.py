from sys import argv
from lib.datasets import load_magic_hat
from lib.ft_statistics import ft_mean
from lib.preprocessing import MinMaxScaler
from lib.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def show_train_history(train_history):
    plt.figure('train history')
    for hist in train_history.values():
        plt.plot(hist)
    plt.legend(train_history.keys())
    plt.xlabel('iteration')
    plt.ylabel('cost')
    plt.show()


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
        solver_list = ['gd', 'sgd', 'mbgd']
        lr = {solver_name: LogisticRegression(solver=solver_name, verbose=1)
              .fit(X_train, y_train) for solver_name in solver_list}
        show_train_history({solver_name: solver.get_fit_history()
                            for solver_name, solver in lr.items()})
    except (Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == '__main__':
    main()
