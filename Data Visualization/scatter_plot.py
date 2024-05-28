import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# python Data\ Visualization/scatter_plot.py dataset_train.csv
# What are the two features that are similar?
def show_correlation_heatmap(corr):
    plt.figure(figsize=(12, 10), num='Correlation Heatmap')
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of All Features")
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.4)  # Adjust left and right margins
    plt.show()

def show_multiple_scatter_plots(df, high_corr_pairs):
    if not high_corr_pairs:
        print("No pairs with correlation >= 0.8 found.")
        return

    num_pairs = len(high_corr_pairs)
    cols = 3
    rows = (num_pairs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), num='What are the two features that are similar?')
    fig.suptitle('What are the two features that are similar?', fontsize=16)
    axes = axes.flatten()

    colors = ["Pink", "Wheat", "Lightskyblue", "Darkseagreen"]
    houses = df["Hogwarts House"].unique()

    for i, (course1, course2) in enumerate(high_corr_pairs):
        ax = axes[i]
        for house, color in zip(houses, colors):
            house_data = df[df["Hogwarts House"] == house]
            ax.scatter(
                house_data[course1],
                house_data[course2],
                alpha=0.5,
                color=color,
                label=house,
                s=50,
            )
        ax.set_title(f"Scatter Plot of {course1} vs {course2}")
        ax.set_xlabel(course1)
        ax.set_ylabel(course2)
        ax.legend(title="Hogwarts House")
        ax.grid(True)

    # Hide any unused subplots
    for i in range(num_pairs, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
    print("Scatter plots saved as 'hogwarts_scatterplot.png'.")

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
        if abs(cor) >= 0.8
    ]
    filtered_correlations.sort(reverse=True, key=lambda x: x[0])

    # Extract the high correlation pairs
    high_corr_pairs = [
        (correlation_matrix.columns[idx1], correlation_matrix.columns[idx2])
        for _, idx1, idx2 in filtered_correlations
    ]

    # Display sorted results
    for cor, idx1, idx2 in filtered_correlations:
        course1, course2 = (
            correlation_matrix.columns[idx1],
            correlation_matrix.columns[idx2],
        )
        print(f"[ {cor:.2f} ] \t {course1} vs. {course2}")

    # Show scatter plots for high correlation pairs
    show_multiple_scatter_plots(df, high_corr_pairs)

if __name__ == "__main__":
    main()
