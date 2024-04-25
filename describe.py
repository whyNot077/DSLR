import sys
import pandas as pd
import ft_statistics as stat

# python describe.py dataset_train.csv
# df = pd.DataFrame(data)
# describe_output = df.describe()

def main():
    if len(sys.argv) != 2:
        print("[Usage] describe.py dataset_train.csv")
        return
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    numeric_columns = df.select_dtypes(include=['int', 'float'])
    numeric_columns = numeric_columns.drop('Index', axis=1)

    result = pd.DataFrame(index=numeric_columns.columns)
    result['Count'] = numeric_columns.apply(stat.ft_count)
    result['Mean'] = numeric_columns.apply(stat.ft_mean)
    for column in numeric_columns.columns:
        numeric_columns[column] = [x if not pd.isna(x) else result.at[column, 'Mean'] for x in numeric_columns[column]]

    result['Std'] = numeric_columns.apply(stat.ft_std)
    result['Min'] = numeric_columns.apply(stat.ft_min)
    result['25%'] = numeric_columns.apply(lambda col: stat.ft_percentile(col, 25))
    result['50%'] = numeric_columns.apply(lambda col: stat.ft_percentile(col, 50))
    result['75%'] = numeric_columns.apply(lambda col: stat.ft_percentile(col, 75))
    result['Max'] = numeric_columns.apply(stat.ft_max)

    show_df(result.transpose())


def show_df(df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)
    

if __name__ == "__main__":
    main()
