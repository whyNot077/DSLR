import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# From this visualization, what features are you going to use for your logistic regression?
# python box_plot.py dataset_train.csv
def show_box_plot(data):
    house_colors = {
        "Gryffindor": "Pink",
        "Slytherin": "Darkseagreen",
        "Ravenclaw": "Lightskyblue",
        "Hufflepuff": "Wheat",
    }

    numeric_columns = [col for col in data.columns if col != "Hogwarts House"]
    num_columns = len(numeric_columns)

    columns_per_row = 4
    rows = (num_columns + columns_per_row - 1) // columns_per_row

    fig, axes = plt.subplots(
        rows, columns_per_row, figsize=(20, 5 * rows), squeeze=False
    )

    for i, column in enumerate(numeric_columns):
        row = i // columns_per_row
        col = i % columns_per_row
        ax = axes[row][col]
        sns.boxplot(
            x="Hogwarts House",
            y=column,
            hue="Hogwarts House",
            data=data,
            ax=ax,
            palette=house_colors,
            dodge=False,
        )
        ax.set_title(f"Box Plot of {column}")

    for j in range(i + 1, rows * columns_per_row):
        axes[j // columns_per_row][j % columns_per_row].axis("off")

    plt.tight_layout()
    plt.savefig("hogwarts_boxplot.png", format="png", dpi=300)
    print("Box plot saved as 'hogwarts_boxplot.png'.")


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
    show_box_plot(data)


if __name__ == "__main__":
    main()
