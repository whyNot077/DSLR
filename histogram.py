import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#'Astronomy', 'Herbology', 'Ancient Runes'
def show_all(df, courses):
    fig, axs = plt.subplots(3, 5, figsize=(10, 8))
    colors = ['Lightpink', 'Darkseagreen', 'Wheat', 'Lightskyblue']
    houses = ['Gryffindor', 'Slytherin', 'Hufflepuff', 'Ravenclaw']
    bins = np.linspace(-4, 4, 20)

    for i, course in enumerate(courses):
        ax = axs[i // 5, i % 5]
        for color, house in zip(colors, houses):
            scores = df[df['Hogwarts House'] == house][course].dropna()
            standardized_data = (scores - scores.mean()) / scores.std()
            ax.hist(standardized_data, bins=bins, alpha=0.5, color=color, label=house)
        
        ax.set_title(course, fontsize=9)
        ax.set_xlabel('σ', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        if i % 5 == 0:
            ax.set_ylabel('Frequency', fontsize=8)

    for j in range(i + 1, 15):
        axs[j // 5, j % 5].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("[Usage] histogram.py dataset_train.csv")
        return

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination',
               'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
               'Care of Magical Creatures', 'Charms', 'Flying']

    show_all(df, courses)

if __name__ == "__main__":
    main()
