import sys
import pandas as pd
import ft_statistics as stat
import matplotlib.pyplot as plt
import numpy as np

# python histogram.py dataset_train.csv
# Which Hogwarts course has a homogeneous score distribution between all four houses?

def show_histograms(df):
    plt.figure(figsize=(10, 8))
    colors = ['Lightskyblue', 'Darkseagreen', 'Wheat']
    bins = np.arange(-4, 5, 1)

    for idx, column in enumerate(df.columns):
        standardized_data = (df[column] - stat.ft_mean(df[column])) / stat.ft_std(df[column])
        plt.hist(standardized_data, bins=bins, alpha=0.5, color=colors[idx % len(colors)], label=column)

    plt.title('Standard Normal Distribution Histograms')
    plt.xlabel('Standard Deviations (σ)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def show_histograms_by_house(df, course):
    plt.figure(figsize=(10, 8))
    colors = ['Pink', 'Wheat', 'Lightskyblue', 'Darkseagreen']
    course_data_by_house = df.groupby('Hogwarts House')[course]

    for (house, scores), color in zip(course_data_by_house, colors):
        standardized_scores = (scores - stat.ft_mean(scores)) / stat.ft_std(scores)
        plt.hist(standardized_scores, bins=np.arange(-4, 5, 1), alpha=0.5, color=color, label=f"{house}")

    plt.title(f'Standard Normal Distribution Histogram of {course} by House')
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

    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['int', 'float']).drop('Index', axis=1, errors='ignore')

    # Get standard deviations
    std_devs = numeric_columns.apply(lambda col: stat.ft_std(col.dropna()))

    # Select the three smallest standard deviation columns
    most_uniform_courses = std_devs.nsmallest(3).index.tolist()

    # Select data for the most uniform courses
    selected_data = numeric_columns[most_uniform_courses]
    show_histograms(selected_data)

    # Select the course with the smallest standard deviation and show its scores by house
    smallest_std_course = most_uniform_courses[0]
    show_histograms_by_house(df, smallest_std_course)

if __name__ == "__main__":
    main()
