import sys
import pandas as pd
import ft_statistics as stat

# python diff.py dataset_train.csv > b
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
    show_df(numeric_columns.describe())

def show_df(df):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)
    

if __name__ == "__main__":
    main()
