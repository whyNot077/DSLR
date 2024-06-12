from sys import argv
from lib.datasets import load_magic_hat, get_house_list
from lib.preprocessing import MinMaxScaler
from lib.linear_model import LogisticRegression
import pandas as pd
import pickle


def main():
    if len(argv) != 3:
        print(f'[Usage] python3 {argv[0]} dataset_test.csv weights.p')
        return
    try:
        X_test, y_test = load_magic_hat(argv[1])
        with open(argv[2], 'rb') as f:
            column_mean, scaler_params, lr_params = pickle.load(f)
        X_test = X_test.fillna({X_test.columns[i]: column_mean[i]
                                for i in range(len(column_mean))})
        scaler = MinMaxScaler().set_params(scaler_params)
        X_test = scaler.transform(X_test)
        lr = LogisticRegression(verbose=1).set_params(lr_params)
        house_list = get_house_list()
        y_test['Hogwarts House'] = pd.Series(lr.predict(X_test)) \
            .map(lambda idx: house_list[idx])
        y_test.to_csv('houses.csv', index=False)
    except (Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == '__main__':
    main()
