import pytest

from numpy import isclose

from PyRKHSstats.combinatorics_utilities import n_permute_m, \
    ordered_combinations, check_permutation_strict, \
    generate_strict_permutations


@pytest.mark.parametrize(
    "m,n",
    [
        (11, 12),
        (10, 0),
    ]
)
def test_undefined_number_of_permutations(m, n):
    """
    Test that the function giving the number of arrangements of n items from m
    objects crashes if n > m.
    """
    with pytest.raises(AssertionError):

        n_permute_m(m, n)


@pytest.mark.parametrize(
    "m,n,expected,precision",
    [
        (1, 1, 1, 0.1),
        (8, 7, 40320, 0.01),
        (8, 1, 8, 0.01),
        (10, 5, 30240, 0.01),
    ]
)
def test_n_permute_m(m, n, expected, precision):

    actual = n_permute_m(m, n)

    isclose(expected, actual, atol=precision)


@pytest.mark.parametrize(
    "m,n,expected",
    [
        (3,
         1,
         [(0,), (1,), (2,)]),
        (3,
         2,
         [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]),
        (3,
         3,
         [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]),
    ]
)
def test_ordered_combinations(m, n, expected):

    actual = ordered_combinations(m, n)

    assert set(expected) == set(actual)


@pytest.mark.parametrize(
    "indices,expected",
    [
        ([3, 2, 1, 0], True),
        ([1, 0, 3, 2], True),
        ([0, 1, 2, 3], False),
        ([0, 3, 2, 1], False),
    ]
)
def test_check_permutation_strict(indices, expected):

    assert check_permutation_strict(indices=indices) == expected


@pytest.mark.parametrize(
    "indices_to_permute,nb_permutations_wanted",
    [
        (list(range(100)), 100),
        (list(range(100)), 200),
        (list(range(100)), 500),
        (list(range(100)), 1000),
        (list(range(100)), 10000),
        (list(range(200)), 1000),
        (list(range(400)), 1000),
    ]
)
def test_generate_strict_permutations(indices_to_permute,
                                      nb_permutations_wanted):

    permutations = generate_strict_permutations(
        indices_to_permute=indices_to_permute,
        nb_permutations_wanted=nb_permutations_wanted
    )

    assert all([check_permutation_strict(permutation) for permutation in
                permutations])
