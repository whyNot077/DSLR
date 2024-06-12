from sys import argv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_correlation_heatmap(corr):
    plt.figure('Correlation Heatmap', figsize=(12, 10))
    plt.title('Correlation Heatmap of All Features', fontsize=16)
    sns.heatmap(corr, cmap='coolwarm', annot=True,
                fmt='.2f', linewidths=0.5, square=True)
    plt.tight_layout(pad=2.4, rect=(0, 0, 0.95, 0.95))
    plt.show()


def show_multiple_scatter_plots(df, high_corr_pairs):
    question = 'What are the two features that are similar?'
    num_pairs = len(high_corr_pairs)
    rows = int(num_pairs ** 0.5)
    cols = (num_pairs + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows),
                             num=question)
    fig.suptitle(question, fontsize=16)
    if type(axes) is np.ndarray:
        axes = axes.flatten()

    colors = ['Pink', 'Wheat', 'Lightskyblue', 'Darkseagreen']
    houses = df['Hogwarts House'].unique()
    for ax, (course1, course2) in zip(axes, high_corr_pairs):
        for house, color in zip(houses, colors):
            house_data = df[df['Hogwarts House'] == house]
            ax.scatter(
                house_data[course1],
                house_data[course2],
                s=50,
                c=color,
                alpha=0.5,
                label=house
            )
        ax.set_title(f'{course1} vs {course2}')
        ax.set_xlabel(course1)
        ax.set_ylabel(course2)
        ax.legend(title='Hogwarts House')
        ax.grid(True)

    for i in range(num_pairs, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=2.4)
    plt.show()


def get_filtered_correlations(correlation_matrix, min_abs_corr):
    upper_triangle_indices = np.triu_indices_from(correlation_matrix, k=1)
    correlations = correlation_matrix.values[upper_triangle_indices]

    filtered_correlations = [
        (corr, (correlation_matrix.columns[idx1],
                correlation_matrix.columns[idx2]))
        for (corr, idx1, idx2)
        in zip(correlations, *upper_triangle_indices)
        if abs(corr) >= min_abs_corr
    ]
    filtered_correlations.sort(reverse=True, key=lambda x: abs(x[0]))
    return filtered_correlations


def main():
    if len(argv) < 2 or len(argv) > 3:
        print(f'[Usage] {argv[0]} dataset_train.csv [min_abs_corr]')
        return
    try:
        csv_file = argv[1]
        df = pd.read_csv(csv_file)
        df = df.drop('Index', axis=1, errors='ignore')
        numeric_columns = df.select_dtypes(include=np.number)
        min_abs_corr = 0.8 if len(argv) == 2 else float(argv[2])
        assert 0 <= min_abs_corr <= 1, 'min_abs_corr must be between 0 and 1.'

        correlation_matrix = numeric_columns.corr()
        show_correlation_heatmap(correlation_matrix)

        filtered_correlations = get_filtered_correlations(correlation_matrix,
                                                          min_abs_corr)
        assert filtered_correlations, \
            f'No pairs with correlation >= {min_abs_corr} were found.'
        for corr, (course1, course2) in filtered_correlations:
            print(f'[ {corr:+.2f} ]\t{course1} vs {course2}')

        high_corr_pairs = [pair for _, pair in filtered_correlations]
        show_multiple_scatter_plots(df, high_corr_pairs)
    except (Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == '__main__':
    main()
