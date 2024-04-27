import sys
import pandas as pd
import ft_statistics as stat
import matplotlib.pyplot as plt
import sys
import pandas as pd
import ft_statistics as stat
import matplotlib.pyplot as plt
import numpy as np

# python histogram.py dataset_train.csv
# Which Hogwarts course has a homogeneous score distribution between all four houses?
def show_histograms(data):
    plt.figure(figsize=(10, 8))
    colors = ['Lightskyblue', 'Darkseagreen', 'Wheat']
    bins = np.arange(-4, 5, 1)

    for idx, (title, column_data) in enumerate(data.items()):
        standardized_data = (column_data - stat.ft_mean(column_data)) / stat.ft_std(column_data)
        plt.hist(standardized_data, bins=bins, alpha=0.5, color=colors[idx % len(colors)], label=title)

    plt.title('Standard Normal Distribution Histograms')
    plt.xlabel('Standard Deviations (σ)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("[Usage] histogram.py dataset_train.csv")
        return

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    numeric_columns = df.select_dtypes(include=['int', 'float'])
    numeric_columns = numeric_columns.drop('Index', axis=1, errors='ignore')

    std_devs = {column: stat.ft_std(numeric_columns[column].dropna()) for column in numeric_columns.columns}
    most_uniform_courses = sorted(std_devs, key=std_devs.get)[:3]

    selected_data = {course: numeric_columns[course] for course in most_uniform_courses}
    show_histograms(selected_data)

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

    selected_columns_dict = {label: numeric_columns[label] for label in selected_label}
    show_histograms(selected_columns_dict)

if __name__ == "__main__":
    main()
