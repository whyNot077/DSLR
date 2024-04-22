import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from itertools import combinations

# python prediction.py
def process_birthday(data):
    if 'Birthday' in data.columns:
        base_date = pd.to_datetime('1996-01-01')
        data['Birthday'] = (pd.to_datetime(data['Birthday']) - base_date).dt.days
    return data

def process_besthand(data):
    if 'Best Hand' in data.columns:
        data = pd.get_dummies(data, columns=['Best Hand'], drop_first=True, dtype='int64')
    return data

def logistic_regression(df, columns_to_keep, house):
    # Select and process the columns
    data = df[columns_to_keep].copy()
    data = process_birthday(data)
    data = process_besthand(data)

    # Handle missing values: fill with mean
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_imputed = imputer.fit_transform(data)

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(df_imputed)

    final_result = 0
    for _ in range(100):
        train_X, test_X, train_Y, test_Y = train_test_split(data_normalized, house, test_size=0.3, random_state=42)
        lr = LogisticRegression(max_iter=100, multi_class='ovr')
        lr.fit(train_X, train_Y)
        predicted_house = lr.predict(test_X)
        final_result += accuracy_score(test_Y, predicted_house) 

    print(f"{final_result}%")
    return final_result

# python prediction.py
if __name__ == "__main__":
    csv_file = 'dataset_train.csv'
    df = pd.read_csv(csv_file)
    house = df['Hogwarts House']

    # all prediction
    initial_columns = [col for col in df.columns if col not in ['Hogwarts House', 'Index', 'First Name', 'Last Name']]
    print("Initial columns")
    logistic_regression(df, initial_columns, house)
    
    # Feature selection
    # sbs_results = sbs_algorithm(df, initial_columns, house)

    # Selected prediction
    columns_to_keep = ['Astronomy', 'Herbology', 'Ancient Runes']
    print("Selected Features : Astronomy, Herbology, Ancient Runes")
    logistic_regression(df, columns_to_keep, house)