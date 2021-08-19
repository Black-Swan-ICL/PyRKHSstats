import pytest
import numpy as np

from math import log, sqrt
from sklearn.gaussian_process.kernels import RBF, DotProduct

from PyRKHSstats.kernel_wrapper import KernelWrapper


observations_1 = np.asarray([[1], [1], [sqrt(2 * log(2)) + 1]])
gram_matrix_1 = np.asarray([[1., 1., 0.5], [1., 1., 0.5], [0.5, 0.5, 1.]])
observations_2 = np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
gram_matrix_2 = np.ones((3, 3))

observations_3 = np.asarray([[1], [2], [3], [4]])
observations_4 = np.asarray([[5], [6]])
rectangular_kernel_matrix = np.asarray([[5, 6], [10, 12], [15, 18], [20, 24]])


def rbf(x, y):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    res = np.exp(
        - (1 / 2) * np.linalg.norm(x - y) ** 2
    )

    return res


def dot_product(x, y):

    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    return np.transpose(x) @ y


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
    "kernel, data, expected, precision",
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


@pytest.mark.parametrize(
    "kernel, x, y, expected, precision",
    [
        (
                DotProduct(sigma_0=0),
                observations_3,
                observations_4,
                rectangular_kernel_matrix,
                0.01
        ),
        (
                dot_product,
                observations_3,
                observations_4,
                rectangular_kernel_matrix,
                0.01
        ),
    ]
)
def test_compute_rectangular_kernel_matrix(kernel, x, y, expected, precision):

    kernel = KernelWrapper(kernel)
    actual = kernel.compute_rectangular_kernel_matrix(x, y)

    assert np.allclose(actual, expected, precision)
