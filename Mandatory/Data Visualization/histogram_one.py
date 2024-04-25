import sys
import pandas as pd
import numpy as np
import ft_statistics as stat
import matplotlib.pyplot as plt

# python histogram.py dataset_train.csv
# Which Hogwarts course has a homogeneous score distribution between all four houses?
def show_histogram(selected_columns):
    plt.figure(figsize=(10, 6))
    plt.hist(selected_columns.dropna(), bins=10, color='Lightskyblue')
    plt.title(f'Histogram of {selected_columns.name}')
    plt.xlabel(f'{selected_columns.name}')
    plt.ylabel('Frequency')
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("[Usage] describe.py dataset_train.csv")
        return
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    numeric_columns = df.select_dtypes(include=['int', 'float'])
    numeric_columns = numeric_columns.drop('Index', axis=1)

    column_labels = numeric_columns.columns.tolist()
    print("================== The Hogwarts Course ==================")
    for i, label in enumerate(column_labels):
        print(f"{i+1}. {label}")
    print("=========================================================")
    selected_label_index = int(input("Choose the course number: ")) - 1
    selected_label = column_labels[selected_label_index]
    
    selected_columns = numeric_columns.loc[:, selected_label]
    selected_columns = selected_columns.fillna(stat.ft_mean(selected_columns))
    show_histogram(selected_columns)


if __name__ == "__main__":
    main()
