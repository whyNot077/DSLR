import sys
import pandas as pd
import numpy as np
import ft_statistics as stat
import matplotlib.pyplot as plt

# python histogram.py dataset_train.csv
# Which Hogwarts course has a homogeneous score distribution between all four houses?
def show_histogram(selected_columns):
    colors = ['Lightskyblue', 'Darkseagreen', 'Wheat']
    plt.figure(figsize=(10, 6))
    bins = np.arange(-4, 5, 1)
    for idx, column in enumerate(selected_columns.columns):
        data = selected_columns[column].dropna()
        standardized_data = (data - stat.ft_mean(data)) / data.std()

        plt.hist(standardized_data, bins=bins, alpha=0.5, color=colors[idx % len(colors)], label=column)
    
    plt.title('Standard Normal Distribution Histograms of Selected Columns')
    plt.xlabel('Standard Deviations (σ)')
    plt.ylabel('Frequency')
    plt.legend()
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
    selected_label = []
    for _ in range(3):
        selected_label_index = int(input("\nChoose the course number: ")) - 1
        selected = column_labels[selected_label_index]
        if selected not in selected_label:
            selected_label.append(selected)
        print(f"\n << {selected} >> is selected")
    selected_columns = numeric_columns.loc[:, selected_label]

    for column in selected_columns.columns:
        mean = stat.ft_mean(selected_columns[column])
        selected_columns[column] = selected_columns[column].fillna(mean)

    show_histogram(selected_columns)


if __name__ == "__main__":
    main()
