from sys import argv
import numpy as np
import pandas as pd
import lib.ft_statistics as stat


def ft_describe(df):
    df = df.select_dtypes(include=np.number)
    result = pd.DataFrame(index=df.columns)
    result['Count'] = df.apply(stat.ft_count)
    result['Mean'] = df.apply(stat.ft_mean)
    result['Std'] = df.apply(stat.ft_std)
    result['Min'] = df.apply(stat.ft_min)
    result['25%'] = df.apply(lambda col: stat.ft_quantile(col, 0.25))
    result['50%'] = df.apply(lambda col: stat.ft_quantile(col, 0.50))
    result['75%'] = df.apply(lambda col: stat.ft_quantile(col, 0.75))
    result['Max'] = df.apply(stat.ft_max)
    result = result.transpose()
    return result


def show_df(df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)


def main():
    if len(argv) != 2:
        print(f'[Usage] python3 {argv[0]} dataset_train.csv')
        return
    try:
        file_name = argv[1]
        df = pd.read_csv(file_name).drop('Index', axis=1, errors='ignore')
        show_df(ft_describe(df))
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
