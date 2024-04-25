import sys
import pickle
from datasets import load_magic_hat, get_house_list
from preprocessing import MinMaxScaler
from linear_model import LogisticRegression
import pandas as pd


def main():
    if len(sys.argv) != 2:
        print('[Usage] python3', sys.argv[0], 'dataset_test.csv')
        return
    try:
        X_test, y_test = load_magic_hat(sys.argv[1])
        with open('weights.p', 'rb') as f:
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
