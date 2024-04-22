import pandas as pd

# python show.py
# ex : csv_file = 'dataset_train.csv'

def show_ssc(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        assert False, 'Cannot read csv file'
    pd.set_option('display.max_columns', None)  # 모든 열 표시
    # pd.set_option('display.max_rows', None)     # 모든 행 표시
    print(df)