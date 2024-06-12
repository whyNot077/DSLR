from sys import argv
import pandas as pd
import matplotlib.pyplot as plt


def show_histogram_by_house(df, course):
    title = f'Histogram of {course} by house'
    plt.figure(title, figsize=(10, 8))
    course_data_by_house = df.groupby('Hogwarts House')[course]
    houses = ['Hufflepuff', 'Ravenclaw', 'Gryffindor', 'Slytherin']
    colors = ['Pink', 'Wheat', 'Lightskyblue', 'Darkseagreen']
    for house, color in zip(houses, colors):
        plt.hist(
            course_data_by_house.get_group(house),
            bins=20,
            color=color,
            label=house
        )
    plt.title(title, fontsize=16)
    plt.xlabel(course)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def main():
    if len(argv) < 2 or len(argv) > 3:
        print(f'[Usage] {argv[0]} dataset_train.csv [course]')
        return
    course = 'Care of Magical Creatures' if len(argv) == 2 else argv[2]
    try:
        csv_file = argv[1]
        df = pd.read_csv(csv_file)
        show_histogram_by_house(df, course)
    except (Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == '__main__':
    main()
