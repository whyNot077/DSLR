import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# From this visualization, what features are you going to use for your logistic regression?
# python pair_plot.py dataset_train.csv
def show_pair_plot(data, name):
    sns.set_theme(style="whitegrid")

    num_features = len(data.columns) - 1
    size_factor = 2
    min_size = 12
    max_size = 24
    figsize = min(max(size_factor * num_features, min_size), max_size)
    
    plt.figure(figsize=(figsize, figsize), num='what features are you going to use for your logistic regression?')
    plt.suptitle('what features are you going to use for your logistic regression?', fontsize=16)
    
    house_colors = {
        "Gryffindor": "Pink",
        "Slytherin": "Darkseagreen",
        "Ravenclaw": "Lightskyblue",
        "Hufflepuff": "Wheat",
    }

    g = sns.pairplot(
        data,
        hue="Hogwarts House",
        palette=house_colors,
        diag_kind="hist",
        markers="o",
        plot_kws={"alpha": 0.6, "s": 30},
        diag_kws={"alpha": 0.7},
    )

    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_ylabel(ax.get_ylabel(), rotation=0)
            ax.xaxis.labelpad = 30
            ax.yaxis.labelpad = 90

    plt.subplots_adjust(left=0.19, bottom=0.15)
    plt.savefig(name, format="png", dpi=300)
    print(f"Pair plot saved as {name}.png")


def main():
    if len(sys.argv) != 2:
        print("Usage: python script_name.py dataset_filename.csv")
        return
    filename = sys.argv[1]

    data = pd.read_csv(filename)
    numeric_columns = data.select_dtypes(include=["number"])
    numeric_columns = numeric_columns.drop("Index", axis=1, errors="ignore")
    data = pd.concat([data["Hogwarts House"], numeric_columns], axis=1)
    
    show_pair_plot(data, 'pairplot.png')

    X = data[['Astronomy', 'Herbology', 'Divination',
              'Muggle Studies', 'Potions', 'Flying', 'Hogwarts House']]
    show_pair_plot(X, 'choosed.png')


if __name__ == "__main__":
    main()
