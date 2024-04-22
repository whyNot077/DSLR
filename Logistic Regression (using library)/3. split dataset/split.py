import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# python split.py
csv_file = 'dataset_train.csv'
df = pd.read_csv(csv_file)
# print(df) # [1600 rows x 19 columns]

# Remove Index, FirstName, LastName
df.drop(['Index', 'First Name', 'Last Name'], axis=1, inplace=True)
base_date = pd.to_datetime('1996-01-01')
df['Birthday'] = (pd.to_datetime(df['Birthday']) - base_date).dt.days
# print(df['Birthday'])

# One-hot-encoding for Best Hand and Hogwarts House
# print(df['Best Hand'].isnull().sum())
df = pd.get_dummies(df, columns=['Best Hand'], drop_first=True, dtype='int64')
# print(df['Hogwarts House'].isnull().sum())
house = pd.get_dummies(df['Hogwarts House'], prefix_sep=' ', dtype='int64')
df = df.drop('Hogwarts House', axis=1)
df = pd.concat([df, house], axis=1)
# print(df['Gryffindor'], df['Hufflepuff'], df['Ravenclaw'], df['Slytherin'])

# Handle missing values : fill with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_imputed = imputer.fit_transform(df)

# Normalize the data
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df_imputed)
# print(df_imputed)

# Split the dataset into training and test
X_train, X_test = train_test_split(df_normalized, 
                                   test_size=0.3, random_state=42)

print(df_normalized.shape)
print(X_train.shape)
print(X_test.shape)