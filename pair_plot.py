from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def save_pair_plot(data, file_name):
    house_colors = {
        'Gryffindor': 'Pink',
        'Hufflepuff': 'Wheat',
        'Ravenclaw': 'Lightskyblue',
        'Slytherin': 'Darkseagreen'
    }

    sns.set_theme(style='whitegrid')
    g = sns.pairplot(data,
                     hue='Hogwarts House',
                     palette=house_colors,
                     diag_kind='hist',
                     aspect=1.11,
                     plot_kws={'s': 10, 'alpha': 0.6, 'linewidth': 0.1},
                     diag_kws={'bins': 20, 'alpha': 0.6},
                     grid_kws={'diag_sharey': False})

    for lh in g._legend.legend_handles:
        lh.set_markersize(10)
        lh.set_alpha(1)

    g.figure.align_ylabels(g.axes.flatten())
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_ylabel(ax.get_ylabel(), rotation=0)
            ax.xaxis.labelpad = 30
            ax.yaxis.labelpad = 60
    g.tight_layout()

    plt.savefig(file_name, dpi=200, format='png')
    print(f'Pair plot saved as {file_name}')


def main():
    if len(argv) != 2:
        print(f'[Usage] python3 {argv[0]} dataset_train.csv')
        return
    try:
        file_name = argv[1]
        data = pd.read_csv(file_name).drop('Index', axis=1, errors='ignore')
        numeric_columns = data.select_dtypes(include=np.number)
        data = pd.concat([data['Hogwarts House'], numeric_columns], axis=1)

        choosed = data[['Hogwarts House', 'Astronomy', 'Herbology',
                        'Divination', 'Muggle Studies', 'Potions', 'Flying']]
        save_pair_plot(choosed, 'pair_plot_choosed.png')
        save_pair_plot(data, 'pair_plot_all.png')
    except (Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == '__main__':
    main()
