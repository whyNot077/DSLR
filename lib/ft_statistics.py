from functools import reduce
import numpy as np
import pandas as pd


def ft_dropna(iterable):
    return list(filter(pd.notna, iterable))


def ft_sum(iterable, skipna=True):
    if skipna:
        iterable = ft_dropna(iterable)
    return reduce(lambda x, y: x + y, iterable, 0)


def ft_count(iterable, skipna=True):
    if skipna:
        iterable = ft_dropna(iterable)
    return len(iterable)


def ft_min(arg1, arg2=None, skipna=True):
    if arg2 is not None:
        return arg2 if arg1 > arg2 else arg1
    iterable = ft_dropna(arg1) if skipna else arg1
    if not len(iterable):
        return np.nan
    return reduce(lambda x, y: ft_min(x, y), iterable)


def ft_max(arg1, arg2=None, skipna=True):
    if arg2 is not None:
        return arg2 if arg1 < arg2 else arg1
    iterable = ft_dropna(arg1) if skipna else arg1
    if not len(iterable):
        return np.nan
    return reduce(lambda x, y: ft_max(x, y), iterable)


def ft_mean(iterable, skipna=True):
    if skipna:
        iterable = ft_dropna(iterable)
    if not len(iterable):
        return np.nan
    return ft_sum(iterable, False) / len(iterable)


def ft_var(iterable, skipna=True):
    if skipna:
        iterable = ft_dropna(iterable)
    if len(iterable) < 2:
        return np.nan
    mean = ft_mean(iterable, False)
    var = ft_sum(map(lambda x: (x - mean) ** 2, iterable), False)
    var /= (len(iterable) - 1)
    return var


def ft_std(iterable, skipna=True):
    return ft_var(iterable, skipna) ** 0.5


def ft_quantile(iterable, q=0.5, skipna=True):
    if skipna:
        iterable = ft_dropna(iterable)
    if not len(iterable):
        return np.nan
    iterable = sorted(iterable)
    index = q * (len(iterable) - 1)
    lower = int(index)
    higher = ft_min(lower + 1, len(iterable) - 1)
    interpolation = (iterable[higher] - iterable[lower]) * (index - lower)
    return iterable[lower] + interpolation
