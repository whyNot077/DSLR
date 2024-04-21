import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# python select_feature.py
# csv_file = 'dataset_train.csv'
# df = pd.read_csv(csv_file)
# # print(df) # [1600 rows x 19 columns]

# # Remove Index, FirstName, LastName
# df.drop(['Index', 'First Name', 'Last Name'], axis=1, inplace=True)
# base_date = pd.to_datetime('1996-01-01')
# df['Birthday'] = (pd.to_datetime(df['Birthday']) - base_date).dt.days
# # print(df['Birthday'])

# # One-hot-encoding for Best Hand and Hogwarts House
# # print(df['Best Hand'].isnull().sum())
# df = pd.get_dummies(df, columns=['Best Hand'], drop_first=True, dtype='int64')
# # print(df['Hogwarts House'].isnull().sum())
# house = pd.get_dummies(df['Hogwarts House'], prefix_sep=' ', dtype='int64')
# df = df.drop('Hogwarts House', axis=1)
# # df = pd.concat([df, house], axis=1)
# # print(df['Gryffindor'], df['Hufflepuff'], df['Ravenclaw'], df['Slytherin'])

# # Handle missing values : fill with mean
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# df_imputed = imputer.fit_transform(df)

# # Normalize the data
# scaler = StandardScaler()
# data = scaler.fit_transform(df_imputed)
# # print(df_imputed)

# X_train, X_test, Y_train, Y_test = train_test_split(data, house, test_size=0.3, random_state=42)
# print(data.shape)
# print(X_train.shape)
# print(X_test.shape)

# target_name = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
# model = LogisticRegression()

# python test.py
csv_file = 'dataset_train.csv'
df = pd.read_csv(csv_file)

# Remove Index, FirstName, LastName, Hogwarts House
house = df['Hogwarts House']
data = df.drop(['Index', 'First Name', 'Last Name', 'Hogwarts House'], axis=1)
base_date = pd.to_datetime('1996-01-01')
data['Birthday'] = (pd.to_datetime(data['Birthday']) - base_date).dt.days

# One-hot-encoding for Best Hand and Hogwarts House
data = pd.get_dummies(data, columns=['Best Hand'], drop_first=True, dtype='int64')
house = pd.get_dummies(house, prefix_sep=' ', dtype='int64')
for column_name in data.columns:
    data[column_name] = data[column_name].fillna(data[column_name].mean())

scaler = MinMaxScaler().fit(data)
data = scaler.transform(data)

target_name = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

final_result = 0

final_result = 0
for _ in range(100):
    X_train, X_test, Y_train, Y_test = train_test_split(data, house)
    proba = [None] * 4
    for i in range(4):
        lr = LogisticRegression()
        lr.fit(X_train, Y_train[target_name[i]])
        proba[i] = lr.predict_proba(X_train)
        proba[i] = [res[1] for res in proba[i]]

    result = [0] * len(X_train)
    for i in range(len(X_train)):
        for j in range(4):
            if proba[result[i]][i] < proba[j][i]:
                result[i] = j

    count = 0
    for i in range(len(result)):
        if Y_train.iloc[i, result[i]]:
            count += 1
    final_result += count
final_result /= len(X_train)

print(f'{final_result}%')


df.drop(['Care of Magical Creatures', 'Defense Against the Dark Arts', 'History of Magic', 'Flying', 'Muggle Studies',  'Transfiguration', 'Ancient Runes'], axis=1, inplace=True)

# Handle missing values : fill with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_imputed = imputer.fit_transform(data)

# Normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(df_imputed)

X_train, X_test, Y_train, Y_test = train_test_split(data, house, test_size=0.3, random_state=42)


target_name = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
model = LogisticRegression()

final_result = 0
for _ in range(100):
    proba = [None] * 4
    for i in range(4):
        lr = LogisticRegression()
        lr.fit(X_train, Y_train[target_name[i]])
        proba[i] = lr.predict_proba(X_train)
        proba[i] = [res[1] for res in proba[i]]

    result = [0] * len(X_train)
    for i in range(len(X_train)):
        for j in range(4):
            if proba[result[i]][i] < proba[j][i]:
                result[i] = j

    count = 0
    for i in range(len(result)):
        if Y_train.iloc[i, result[i]]:
            count += 1
    final_result += count
final_result /= len(X_train)

print(f'{final_result}%')



df.drop(['Divination', 'Potions', 'Charms'], axis=1, inplace=True)

# Handle missing values : fill with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_imputed = imputer.fit_transform(data)

# Normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(df_imputed)

X_train, X_test, Y_train, Y_test = train_test_split(data, house, test_size=0.3, random_state=42)


target_name = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
model = LogisticRegression()

final_result = 0
for _ in range(100):
    proba = [None] * 4
    for i in range(4):
        lr = LogisticRegression()
        lr.fit(X_train, Y_train[target_name[i]])
        proba[i] = lr.predict_proba(X_train)
        proba[i] = [res[1] for res in proba[i]]

    result = [0] * len(X_train)
    for i in range(len(X_train)):
        for j in range(4):
            if proba[result[i]][i] < proba[j][i]:
                result[i] = j

    count = 0
    for i in range(len(result)):
        if Y_train.iloc[i, result[i]]:
            count += 1
    final_result += count
final_result /= len(X_train)

print(f'{final_result}%')