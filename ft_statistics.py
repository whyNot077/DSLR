import array
from functools import reduce
import pandas as pd
import math

def ft_mean(x):
    filtered_x = [xi for xi in x if not math.isnan(xi)]
    return sum(filtered_x) / len(filtered_x) if filtered_x else float('nan')

def ft_percentile(x, p):
    sorted_x = sorted(x)
    length = len(sorted_x)
    
    pos = (length - 1) * p / 100
    lower = int(pos)
    higher = min(lower + 1, length - 1)
    
    interpolation = (sorted_x[higher] - sorted_x[lower]) * (pos - lower)
    return sorted_x[lower] + interpolation

# pandas var : 불편분산
def ft_var(x: list[int]) -> float:
    mean = ft_mean(x)
    deviation = x - mean
    variance = ft_mean(deviation ** 2)
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
