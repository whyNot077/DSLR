import array
from functools import reduce
import pandas as pd

def ft_mean(x):
    valid_numbers = [num for num in x if not pd.isna(num) and isinstance(num, (int, float))]
    if not valid_numbers:
        return 0
    return sum(valid_numbers) / len(valid_numbers)


def ft_median(x):
    valid_numbers = [num for num in x if not pd.isna(num) and isinstance(num, (int, float))]
    if not valid_numbers:
        return 0 

    sorted_x = sorted(valid_numbers)
    length = len(sorted_x)

    if length % 2 == 0:
        big_i = length // 2
        small_i = big_i - 1
        middle = (sorted_x[big_i] + sorted_x[small_i]) / 2.0
    else:
        middle = sorted_x[length // 2]

    return (float(middle))

def ft_quartile(x):
    valid_numbers = [num for num in x if not pd.isna(num) and isinstance(num, (int, float))]
    if not valid_numbers:
        return [0, 0]
    
    sorted_x = sorted(valid_numbers)
    length = len(sorted_x)
    mid_index = length // 2

    q1 = ft_median(sorted_x[:mid_index])
    q3 = ft_median(sorted_x[mid_index + (length % 2):])
    return ([float(q1), float(q3)])

def ft_percentile(x, p):
    valid_numbers = [num for num in x if not pd.isna(num) and isinstance(num, (int, float))]
    if not valid_numbers or not 0 <= p <= 100:
        return 0
    
    sorted_x = sorted(x)
    length = len(sorted_x)
    
    pos = (length - 1) * p / 100
    lower_index = int(pos)
    upper_index = min(lower_index + 1, length - 1)
    
    if lower_index == upper_index:
        return sorted_x[lower_index]
    else:
        interpolation = (sorted_x[upper_index] - sorted_x[lower_index]) * (pos - lower_index)
        return sorted_x[lower_index] + interpolation

def ft_var(num_list: list[int]) -> float:
    valid_numbers = [num for num in num_list if not pd.isna(num)]
    if not valid_numbers:
        return 0
    
    mean = ft_mean(valid_numbers)
    variance = sum((x - mean) ** 2 for x in valid_numbers) / len(valid_numbers)
    return variance

def ft_std(num_list: list[int]) -> float:
    return ft_var(num_list) ** 0.5

def ft_min(arg1, arg2=None):
    if arg2 is not None:
        return arg1 if arg1 < arg2 else arg2
    return reduce(lambda x, y: ft_min(x, y), arg1)

def ft_max(arg1, arg2=None):
    if arg2 is not None:
        return arg1 if arg1 > arg2 else arg2
    return reduce(lambda x, y: ft_max(x, y), arg1)

def ft_count(df_column):
    count = 0
    
    for value in df_column:
        if not pd.isna(value):
            count += 1
    return count
