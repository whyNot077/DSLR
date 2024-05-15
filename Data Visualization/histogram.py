import sys
import pandas as pd
import ft_statistics as stat
import matplotlib.pyplot as plt
import sys
# python histogram.py dataset_train.csv
# Which Hogwarts course has a homogeneous score distribution between all four houses?
def show_histograms(df):
    plt.figure(figsize=(10, 8))
    colors = ['Lightskyblue', 'Darkseagreen', 'Wheat']
    bins = np.arange(-4, 5, 1)

    for idx, (title, column_data) in enumerate(df.items()):
        standardized_data = (column_data - stat.ft_mean(column_data)) / stat.ft_std(column_data)
        plt.hist(standardized_data, bins=bins, alpha=0.5, color=colors[idx % len(colors)], label=title)

    plt.title('Standard Normal Distribution Histograms')
    plt.xlabel('Standard Deviations (σ)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def show_histograms_by_house(df, course):
    plt.figure(figsize=(10, 8))
    colors = ['Pink', 'Wheat', 'Lightskyblue', 'Darkseagreen']
    course_data_by_house = df.groupby('Hogwarts House')[course]

    for (house, scores), color in zip(course_data_by_house, colors):
        standardized_scores = (scores - stat.ft_mean(scores)) / stat.ft_std(scores)
        plt.hist(standardized_scores, bins=np.arange(-4, 5, 1), alpha=0.5, color=color, label=f"{house}")

    plt.title(f'Standard Normal Distribution Histogram of {course} by House')
    plt.xlabel('Standard Deviations (σ)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("[Usage] histogram.py dataset_train.csv")
        return

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)

    # Select numeric colums
    numeric_columns = df.select_dtypes(include=['int', 'float'])
    numeric_columns = numeric_columns.drop('Index', axis=1, errors='ignore')

    # Get standandard diviations
    std_devs = {column: stat.ft_std(numeric_columns[column].dropna()) for column in numeric_columns.columns}
    most_uniform_courses = sorted(std_devs, key=std_devs.get)[:3]

    # Select the three smallest standard diviation columns.
    selected_data = {course: numeric_columns[course] for course in most_uniform_courses}
    show_histograms(selected_data)

    # Select the course with the smallest standard deviation and show its scores by house.
    smallest_std_course = most_uniform_courses[0]
    show_histograms_by_house(df, smallest_std_course)

if __name__ == "__main__":
    main()
