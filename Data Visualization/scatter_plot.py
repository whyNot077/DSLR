import sys
import pandas as pd
import numpy as np
import ft_statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns


# python scatter_plot.py dataset_train.csv
# What are the two features that are similar ?
def show_correlation_heatmap(corr):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of All Features")
    plt.show()


def show_scatter_plot(df, course1, course2):
    plt.figure(figsize=(10, 8))
    colors = ["Pink", "Wheat", "Lightskyblue", "Darkseagreen"]
    houses = df["Hogwarts House"].unique()

    for house, color in zip(houses, colors):
        house_data = df[df["Hogwarts House"] == house]
        plt.scatter(
            house_data[course1],
            house_data[course2],
            alpha=0.5,
            color=color,
            label=house,
            s=50,
        )

    plt.title(f"Scatter Plot of {course1} vs {course2}")
    plt.xlabel(course1)
    plt.ylabel(course2)
    plt.legend(title="Hogwarts House")
    plt.grid(True)
    plt.show()


def main():
    if len(sys.argv) != 2:
        print("[Usage] describe.py dataset_train.csv")
        return
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    # Select numeric columns
    numeric_columns = df.select_dtypes(include=["int", "float"])
    numeric_columns = numeric_columns.drop("Index", axis=1, errors="ignore")

    # Calculate the correlation matrix and show heatmap
    correlation_matrix = numeric_columns.corr()
    show_correlation_heatmap(correlation_matrix)

    # Use the upper triangle of the correlation matrix to find high correlations
    upper_triangle_indices = np.triu_indices_from(correlation_matrix, k=1)
    correlations = correlation_matrix.values[upper_triangle_indices]

    # Combine correlations with indices and filter by a threshold, then sort by correlation value
    filtered_correlations = [
        (cor, i, j)
        for (cor, i, j) in zip(correlations, *upper_triangle_indices)
        if cor >= 0.7
    ]
    filtered_correlations.sort(reverse=True, key=lambda x: x[0])

    # Display sorted results and scatter plots
    for cor, idx1, idx2 in filtered_correlations:
        course1, course2 = (
            correlation_matrix.columns[idx1],
            correlation_matrix.columns[idx2],
        )
        print(f"[ {cor:.2f} ] \t {course1} vs. {course2}")
        show_scatter_plot(df, course1, course2)


if __name__ == "__main__":
    main()
