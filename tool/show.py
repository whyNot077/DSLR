import pandas as pd

# python show.py
csv_file = 'dataset_train.csv'
df = pd.read_csv(csv_file)
pd.set_option('display.max_columns', None)  # 모든 열 표시
# pd.set_option('display.max_rows', None)     # 모든 행 표시
print(df)