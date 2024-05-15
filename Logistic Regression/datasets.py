import pandas as pd


def get_house_list():
    house_list = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    return house_list


def load_magic_hat(filepath):
    data = pd.read_csv(filepath)
    house_list = get_house_list()
    data['Hogwarts House'] = data['Hogwarts House'] \
        .map(lambda house: house_list.index(house)
             if not pd.isna(house) else house)
    data['Astronomy'] = data['Astronomy'] \
        .fillna(-100 * data['Defense Against the Dark Arts'])
    X = data[['Astronomy', 'Herbology', 'Divination',
              'Muggle Studies', 'Potions', 'Flying']]
    y = data[['Index', 'Hogwarts House']]
    return (X, y)
