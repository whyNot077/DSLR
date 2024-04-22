import pandas as pd
from preprocessing import MinMaxScaler
from linear_model import LogisticRegression
from metrics import accuracy_score

# python tool/lr_test.py
def load_data(filepath):
    data = pd.read_csv(filepath)
    y = data['Hogwarts House'].to_numpy()
    data['Astronomy'] = data['Astronomy'] \
        .fillna(-100 * data['Defense Against the Dark Arts'])
    X = data[['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Potions', 'Flying']]
    # X = data[['Astronomy', 'Herbology', 'Ancient Runes']]
    # X = data.drop(['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1)

    return (X, y)

X_train, y_train = load_data('dataset_train.csv')
X_test, y_test = load_data('dataset_test.csv')
for column in X_train.columns:
    X_train[column] = X_train[column].fillna(X_train[column].mean())
    X_test[column] = X_test[column].fillna(X_train[column].mean())

scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(verbose=1)
lr.fit(X_train, y_train)
predicted = lr.predict(X_test)
score = 100 * accuracy_score(y_test, predicted)
print(f'score = {score:.6f}')
