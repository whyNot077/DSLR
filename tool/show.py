import matplotlib.pyplot as plt
import pandas as pd

# ex : csv_file = 'dataset_train.csv'
def show_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception:
        assert False, 'Cannot read csv file'
    pd.set_option('display.max_columns', None)  # 모든 열 표시
    # pd.set_option('display.max_rows', None)     # 모든 행 표시
    print(df)

def show_df(df):
    pd.set_option('display.max_columns', None)  # 모든 열 표시
    print(df)

def show_graph(data, estimated):
    plt.figure('ft_linear_regression')
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1])
    plt.plot(data.iloc[:, 0], estimated, color='red')
    plt.xlabel('mileage')
    plt.ylabel('price')
    plt.show()
