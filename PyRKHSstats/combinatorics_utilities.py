# TODO reformat docstrings
from itertools import permutations, combinations, chain

import numpy as np


def n_permute_m(m, n):
    """
    Computes the number of arrangement of n items from m objects, mPn, equal to
    m!/(m-n)!.

    Parameters
    ----------
    m : int
        The number of objects.
    n : int
        The number of items.

    Returns
    -------
    float
        mPn, the number of arrangements of n items from m objects.
    """
    assert 0 < n <= m

    return np.cumprod(range(m - n + 1, m + 1))[-1]


def ordered_combinations(m, n):
    """
    Generates a list of all ordered combinations of n elements from a set of m
    objects. For example, if m = 3 and n = 2, the result is
    [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)].

    Parameters
    ----------
    m : int
        The number of objects.
    n : int
        The number of elements.

    Returns
    -------
    list
        A list of all ordered combinations of n elements from m objects.
    """
    l_ordered_combinations = [list(permutations(elt)) for elt in
                              combinations(range(m), n)]
    l_ordered_combinations = list(chain.from_iterable(l_ordered_combinations))

    return l_ordered_combinations
