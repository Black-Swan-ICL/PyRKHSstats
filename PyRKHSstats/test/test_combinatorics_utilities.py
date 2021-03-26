import pytest

from numpy import isclose

from combinatorics_utilities import n_permute_m, ordered_combinations


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
