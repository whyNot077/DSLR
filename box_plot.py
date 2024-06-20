from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_box_plot(data, file_name):
    numeric_columns = data.select_dtypes(include=np.number).columns
    num_columns = len(numeric_columns)
    rows = int(num_columns ** 0.5)
    cols = (num_columns + rows - 1) // rows
    fig_size_factor = 5
    fig_width, fig_height = fig_size_factor * cols, fig_size_factor * rows

    fig, axes = plt.subplots(rows, cols, squeeze=False,
                             figsize=(fig_width, fig_height))
    axes = axes.flatten()
    for i in range(num_columns, len(axes)):
        fig.delaxes(axes[i])

    house_colors = {
        'Gryffindor': 'Pink',
        'Hufflepuff': 'Wheat',
        'Ravenclaw': 'Lightskyblue',
        'Slytherin': 'Darkseagreen'
    }
    for ax, column in zip(axes, numeric_columns):
        sns.boxplot(
            data=data,
            x='Hogwarts House',
            y=column,
            hue='Hogwarts House',
            palette=house_colors,
            ax=ax
        )
        ax.set_title(column)
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(file_name, dpi=200, format='png')
    print(f'Box plot saved as {file_name}')


def main():
    if len(argv) != 2:
        print(f'[Usage] python3 {argv[0]} dataset_train.csv')
        return
    try:
        file_name = argv[1]
        data = pd.read_csv(file_name).drop('Index', axis=1, errors='ignore')
        save_box_plot(data, 'box_plot.png')
    except (Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == '__main__':
    main()
