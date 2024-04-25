import sys
import pandas as pd
import numpy as np
import ft_statistics as stat
import matplotlib.pyplot as plt

# python scatter_plot.py dataset_train.csv
# What are the two features that are similar ?
def show_scatter_plot(class1, class2, feature1, feature2):
    house_colors = {
        'Gryffindor': 'Lightpink',
        'Hufflepuff': 'Wheat',
        'Ravenclaw': 'Lightskyblue',
        'Slytherin': 'Darkseagreen'
    }

    plt.figure(figsize=(10, 6))
    plt.title('Scatter Plot of Two Features')
    plt.xlabel(f'{feature1}')
    plt.ylabel(f'{feature2}')

    plt.scatter(class1[feature1], class1[feature2], \
                color=house_colors[class1['Hogwarts House'].iloc[0]], \
                    alpha=0.8, label=class1['Hogwarts House'].iloc[0])

    plt.scatter(class2[feature1], class2[feature2], \
                color=house_colors[class2['Hogwarts House'].iloc[0]], \
                    alpha=0.8, label=class2['Hogwarts House'].iloc[0])

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
    house = pd.get_dummies(df['Hogwarts House'], prefix_sep=' ', dtype='int64')

    print("================== The Hogwarts House ==================")
    house_labels = house.columns.tolist()
    for i, house_label in enumerate(house_labels):
        print(f"{i+1}.{house_label}")
    print("========================================================")

    house_one_index = int(input("\nChoose The First Hogwart House Number:")) - 1
    house_one = house_labels[house_one_index]
    print(f"\n << {house_one} >> is selected")

    house_two_index = int(input("\nChoose The Second Hogwart House Number:")) - 1
    house_two = house_labels[house_two_index]
    print(f"\n << {house_two} >> is selected")

    while house_two_index == house_one_index:
        print("\nYou already choosed {house_two}.")
        house_two_index = int(input("\nChoose The Second Hogwart House Number:")) - 1
        house_two = house_labels[house_two_index]

    column_labels = numeric_columns.columns.tolist()
    print("================== The Hogwarts Course ==================")
    for i, label in enumerate(column_labels):
        print(f"{i+1}. {label}")
    print("=========================================================")

    selected_label = []
    for _ in range(2):
        selected_label_index = int(input("\nChoose The Hogwart Course Number: ")) - 1
        selected = column_labels[selected_label_index]
        if selected not in selected_label:
            selected_label.append(selected)
        print(f"\n << {selected} >> is selected")
    selected_columns = numeric_columns.loc[:, selected_label]

    for column in selected_columns.columns:
        mean = stat.ft_mean(selected_columns[column])
        selected_columns[column] = selected_columns[column].fillna(mean)

    class_one = pd.concat([selected_columns[house[house_one] == 1], df.loc[house[house_one] == 1, ['Hogwarts House']]], axis=1)
    class_two = pd.concat([selected_columns[house[house_two] == 1], df.loc[house[house_two] == 1, ['Hogwarts House']]], axis=1)
    feature1 = selected_label[0]
    feature2 = selected_label[1]
    show_scatter_plot(class_one, class_two, feature1, feature2)

if __name__ == "__main__":
    main()
