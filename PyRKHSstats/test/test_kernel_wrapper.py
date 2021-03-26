import pytest
import numpy as np

from math import log, sqrt
from sklearn.gaussian_process.kernels import RBF

from kernel_wrapper import KernelWrapper


observations_1 = np.asarray([[1], [1], [sqrt(2 * log(2)) + 1]])
gram_matrix_1 = np.asarray([[1., 1., 0.5], [1., 1., 0.5], [0.5, 0.5, 1.]])
observations_2 = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
gram_matrix_2 = np.ones((3, 3))


def rbf(x, y):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    res = np.exp(
        - (1 / 2) * np.linalg.norm(x - y) ** 2
    )

    return res


@pytest.mark.parametrize(
    "kernel,x,y,expected,precision",
    [
        (RBF(), 1, 1, 1.0, 0.01),
        (RBF(), [1, 1], [1, 1], 1.0, 0.01),
        (rbf, 1, 1, 1.0, 0.01),
        (rbf, [1, 1], [1, 1], 1.0, 0.01),
    ]
)
def test_single_kernel_computation(kernel, x, y, expected, precision):

    my_kernel = KernelWrapper(kernel)

    assert np.isclose(my_kernel.evaluate(x, y), expected, atol=precision)


@pytest.mark.parametrize(
    "kernel,data,expected,precision",
    [
        (RBF(), observations_1, gram_matrix_1, 0.01),
        (RBF(), observations_2, gram_matrix_2, 0.01),
        (rbf, observations_1, gram_matrix_1, 0.01),
        (rbf, observations_2, gram_matrix_2, 0.01),
    ]
)
def test_kernelised_gram_matrix_computation(kernel, data, expected, precision):

    kernel = KernelWrapper(kernel)
    actual = kernel.compute_kernelised_gram_matrix(data)

    assert np.allclose(actual, expected, precision)
