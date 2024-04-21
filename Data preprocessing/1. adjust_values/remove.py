import pandas as pd

# python remove.py
csv_file = 'dataset_train.csv'
df = pd.read_csv(csv_file)

print(df) # [1600 rows x 19 columns]
# print(df.isnull().sum())

new_df = df.dropna(axis=0)
# print(new_df) # [1251 rows x 19 columns]

new_df = df.dropna(axis=1)
print(new_df) # [1600 rows x 8 columns]


