# TODO reformat docstrings
import copy
import random

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


def check_permutation_strict(indices):
    """
    Checks whether a permutation of indices is a strict permutation i.e. that
    :math:`\forall~ i, indices[i] \neq i`.

    Parameters
    ----------
    indices : array_like
        The candidate permutation.

    Returns
    -------
    bool
        Whether the permutation is a strict permutation.
    """
    natural_indices = np.asarray(range(len(indices)))

    return all(np.asarray(indices) != natural_indices)


# TODO add a check on the number of permutations wanted
def generate_strict_permutations(indices_to_permute, nb_permutations_wanted):
    """
    Generates a set of strict permutations i.e. permutations that do not leave
    any element unchanged.

    Parameters
    ----------
    indices_to_permute : array_like
        A sequence we want to permute.
    nb_permutations_wanted : int
        The number of permutations we want to obtain.
    Returns
    -------
    set
        The set of the desired size of strict permutations.
    """
    n = len(indices_to_permute)

    strict_permutations = set()

    while len(strict_permutations) < nb_permutations_wanted:

        candidate_permutation = np.repeat(np.nan, n)

        to_draw_from = list(range(n))

        for i in range(n - 2):

            l = copy.deepcopy(to_draw_from)
            if i in l:
                l.remove(i)
            new_idx = random.sample(l, 1)[0]
            to_draw_from.remove(new_idx)

            candidate_permutation[i] = new_idx

        # Handle the last two elements
        left_penultimate = to_draw_from[0]
        left_last = to_draw_from[1]
        if left_penultimate != (n - 2) and left_last != (n - 1):
            candidate_permutation[n - 2] = left_penultimate
            candidate_permutation[n - 1] = left_last
        else:
            candidate_permutation[n - 2] = left_last
            candidate_permutation[n - 1] = left_penultimate

        strict_permutations.add(tuple(candidate_permutation.astype(int)))

    return strict_permutations
