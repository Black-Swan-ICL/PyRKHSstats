import pytest
import numpy as np

from PyRKHSstats.data_handling import check_conformable, ArraysNonConformable


@pytest.mark.parametrize(
    "args",
    [
        (
            np.asarray([0, 1, 2]),
            np.asarray([3, 4, 5])
        ),
        (
            np.asarray([[1, 2], [3, 4], [5, 6]]),
            np.asarray([[11, 12], [13, 14], [15, 16]]),
            np.asarray([[21, 22], [23, 24], [25, 25]])
        ),
        (
            np.asarray([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[1, 2], [3, 4]]])
        ),
    ]
)
def test_check_conformable(args):

    assert check_conformable(*args)


@pytest.mark.parametrize(
    "args",
    [
        (np.asarray([1, 2, 3, 4]), np.asarray([[1, 2], [3, 4]])),
    ]
)
def test_crash_check_conformable(args):

    with pytest.raises(ArraysNonConformable):
        check_conformable(*args)
