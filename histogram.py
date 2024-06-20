from sys import argv
import pandas as pd
import matplotlib.pyplot as plt


def show_histogram_by_house(data, course):
    title = f'Histogram of {course} by house'
    plt.figure(title, figsize=(10, 8))
    course_data_by_house = data.groupby('Hogwarts House')[course]
    houses = ['Hufflepuff', 'Ravenclaw', 'Gryffindor', 'Slytherin']
    colors = ['Pink', 'Wheat', 'Lightskyblue', 'Darkseagreen']
    for house, color in zip(houses, colors):
        plt.hist(
            course_data_by_house.get_group(house),
            bins=20,
            color=color,
            label=house,
            alpha=0.8
        )
    plt.title(title, fontsize=14)
    plt.xlabel(course)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def main():
    if len(argv) < 2 or len(argv) > 3:
        print(f'[Usage] python3 {argv[0]} dataset_train.csv [course]')
        return
    course = 'Care of Magical Creatures' if len(argv) == 2 else argv[2]
    try:
        file_name = argv[1]
        data = pd.read_csv(file_name)
        show_histogram_by_house(data, course)
    except (Exception, KeyboardInterrupt) as e:
        print(e)


if __name__ == '__main__':
    main()
