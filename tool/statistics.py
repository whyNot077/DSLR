from functools import reduce


def ft_min(arg1, arg2=None):
    if arg2 is not None:
        return arg1 if arg1 < arg2 else arg2
    return reduce(lambda x, y: ft_min(x, y), arg1)


def ft_max(arg1, arg2=None):
    if arg2 is not None:
        return arg1 if arg1 > arg2 else arg2
    return reduce(lambda x, y: ft_max(x, y), arg1)


def ft_mean(iterable):
    return sum(iterable) / len(iterable)
