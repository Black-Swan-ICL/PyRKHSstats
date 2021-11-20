import pytest
import numpy as np

from math import sqrt, log

from sklearn.gaussian_process.kernels import RBF, DotProduct

from PyRKHSstats.hsic import compute_squared_norm_mean_embedding, compute_biased_hsic
from PyRKHSstats.kernel_wrapper import KernelWrapper


@pytest.mark.parametrize(
    "data,kernel,expected,precision",
    [
        (
            np.asarray([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]),
            KernelWrapper(DotProduct(sigma_0=0.)),
            0.,
            0.01
        ),
        (
            np.asarray([2, 2, 2]),
            KernelWrapper(DotProduct(sigma_0=0.)),
            4,
            0.01
        ),
        (
            np.asarray([
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0]
            ]),
            KernelWrapper(RBF()),
            1.,
            0.01
        ),
        (
            np.asarray([1, 1 + sqrt(2*log(2)), 1]),
            KernelWrapper(RBF()),
            2/3,
            0.01
        ),

    ]
)
def test_compute_squared_norm_mean_embedding(data, kernel, expected, precision):

    actual = compute_squared_norm_mean_embedding(data, kernel)

    assert np.isclose(actual, expected, atol=precision)


@pytest.mark.parametrize(
    "data_x,data_y,kernel_k,kernel_l,expec_hsic,expec_K,expec_L,precision",
    [
        (
            np.asarray([
                [1, 0], [2, 2], [2, 3], [0, 1], [-1, 1]
            ]),
            np.asarray([
                [1, 1, 1], [-1, 2, 3], [3, -1, 2], [0, 1, 0], [1, 0, 1]
            ]),
            KernelWrapper(DotProduct(sigma_0=0)),
            KernelWrapper(DotProduct(sigma_0=0)),
            1.648,
            np.asarray([
                [1, 2, 2, 0, -1],
                [2, 8, 10, 2, 0],
                [2, 10, 13, 3, 1],
                [0, 2, 3, 1, 1],
                [-1, 0, 1, 1, 2]
            ]),
            np.asarray([
                [3, 4, 4, 1, 2],
                [4, 14, 1, 2, 2],
                [4, 1, 14, -1, 5],
                [1, 2, -1, 1, 0],
                [2, 2, 5, 0, 2]
            ]),
            0.01
        )
    ]
)
def test_biased_hsic(data_x, data_y, kernel_k, kernel_l, expec_hsic, expec_K,
                     expec_L, precision):

    actual = compute_biased_hsic(data_x=data_x,
                                 data_y=data_y,
                                 kernel_kx=kernel_k,
                                 kernel_ky=kernel_l)

    assert (
            np.isclose(actual['HSIC'], expec_hsic, atol=precision) and
            np.allclose(actual['K'], expec_K, atol=precision) and
            np.allclose(actual['L'], expec_L, atol=precision)
    )

