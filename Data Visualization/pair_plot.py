import sys
import pandas as pd
import numpy as np
import ft_statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns


# From this visualization, what features are you going to use for your logistic regression?
# python pair_plot.py dataset_train.csv
def show_pair_plot(data):
    # Set style and size for the plots
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 16))

    # Define color palette for Hogwarts Houses
    house_colors = {
        "Gryffindor": "Pink",
        "Slytherin": "Darkseagreen",
        "Ravenclaw": "Lightskyblue",
        "Hufflepuff": "Wheat",
    }

    # Create the pairplot
    g = sns.pairplot(
        data,
        hue="Hogwarts House",
        palette=house_colors,
        diag_kind="hist",
        markers="o",
        plot_kws={"alpha": 0.6, "s": 30, "edgecolor": "k"},
        diag_kws={"alpha": 0.7},
    )

    # Rotate labels for better readability
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_ylabel(ax.get_ylabel(), rotation=0)
            ax.xaxis.labelpad = 30
            ax.yaxis.labelpad = 90

    # Adjust layout to make room for rotated labels
    plt.subplots_adjust(left=0.09, bottom=0.05)

    # Save the plot as a file
    plt.savefig("hogwarts_pairplot.png", format="png", dpi=300)
    print("Pair plot saved as 'hogwarts_pairplot.png'.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py dataset_filename.csv")
        return
    filename = sys.argv[1]

    data = pd.read_csv(filename)
    numeric_columns = data.select_dtypes(include=["number"])
    numeric_columns = numeric_columns.drop("Index", axis=1, errors="ignore")
    data = pd.concat([data["Hogwarts House"], numeric_columns], axis=1)
    # print(data)
    show_pair_plot(data)


if __name__ == "__main__":
    main()
