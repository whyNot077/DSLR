import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# python select_feature.py
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
# df = pd.concat([df, house], axis=1)
# print(df['Gryffindor'], df['Hufflepuff'], df['Ravenclaw'], df['Slytherin'])

# Handle missing values : fill with mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df_imputed = imputer.fit_transform(df)

# Normalize the data
scaler = StandardScaler()
data = scaler.fit_transform(df_imputed)
# print(df_imputed)

X_train, X_test, Y_train, Y_test = train_test_split(data, house, test_size=0.3, random_state=42)
print(data.shape)
print(X_train.shape)
print(X_test.shape)

target_name = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
model = LogisticRegression() 

scores = np.zeros(data.shape[1] - 1)
count = np.zeros(data.shape[1] - 1)

for n_features in range(1, data.shape[1]) :
    for i in range(4):
        sfs = SequentialFeatureSelector(model, n_features_to_select=n_features, n_jobs=-1)
        sfs.fit(X_train, Y_train[target_name[i]])  
        f_mask = sfs.support_
        model.fit(X_train[:, f_mask], Y_train[target_name[i]]) 
        scores[n_features - 1] += model.score(X_train[:, f_mask], Y_train[target_name[i]])
        count[n_features - 1] += 1

average_scores = scores / count

plt.plot(range(1, data.shape[1]), average_scores, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of Features')
plt.grid()
plt.tight_layout()
plt.show()

for i in range(4):
    sfs = SequentialFeatureSelector(model, n_features_to_select=5, n_jobs=-1)
    sfs.fit(X_train, Y_train[target_name[i]]) 
    selected_features = df.columns[sfs.get_support()]
    print(f"{target_name[i]} :\n", selected_features)
