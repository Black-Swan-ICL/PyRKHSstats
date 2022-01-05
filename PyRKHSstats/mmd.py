"""
This module contains the code to compute the Maximum Mean Discrepancy (MMD)
following the material presented in papers 'A Fast, Consistent Kernel
Two-Sample Test', A. Gretton, K. Fukumizu, Z. Harchaoui and B. K.
Sriperumbudur (NIPS #22, 2009) and 'A Kernel Two-Sample Test', A. Gretton,
K. M. Borgwardt, M. J. Rasch, B. Sch\"{o}lkopf and A. Smola (Journal of
Machine Learning Research #13, 2012).
"""
import copy
import itertools

import numpy as np

from enum import Enum, unique

from math import sqrt
from scipy.stats import norm


_field_kernel_Gram_Kx = 'mat_kernel_Gram_Kx'
_field_kernel_Gram_Ky = 'mat_kernel_Gram_Ky'
_field_mat_Kxy = 'mat_Kxy'
_field_kernel_Gram_Kxy = 'mat_kernel_Gram_Kxy'


@unique
class ImplementedMMDSchemes(Enum):
    SPECTRAL = 'Gram Matrix Spectrum'
    PERMUTATION = 'Permutation'


class MMDTestingSchemeNotImplemented(Exception):
    """Raised when the user requests a scheme for MMD that is not implemented.
    """


class CaseNotHandled(Exception):
    """Raised when the user attempts to use a method outside of the conditions
    it was planned for.
    """


def compute_unbiased_squared_mmd(data_x, data_y, kernel):
    """
    Computes the unbiased estimate of the squared Maximum Mean Discrepancy
    between :math:`\text{P}_{\text{X}}` and :math:`\text{P}_{\text{Y}}`
    based on the samples of the :math:`x_i`'s and the :math:`y_i`'s.

    Parameters
    ----------
    data_x : array_like
        The :math:`x_i`'s.
    data_y : array_like
        The :math:`y_i`'s.
    kernel : KernelWrapper
        The reproducing kernel associated to the RKHS chosen on the space.

    Returns
    -------
    dic
        A dictionary containing the value of :math:`\text{MMD}_{\text{u}}^2`,
        as well as kernel Gram matrix :math:`\text{K}_{\text{X}}`, kernel Gram
        matrix :math:`\text{K}_{\text{Y}}` and matrix
        :math:`\text{K}_{\text{XY}}`.

    Notes
    -----
    As pointed out by A. Gretton and co-authors in their 2012 'A Kernel
    Two-Sample Test' (paragraph following the proof of lemma 6), computing
    the unbiased estimator of the squared MMD can yield negative values.
    """

    nx = data_x.shape[0]
    ny = data_y.shape[0]

    mat_kernel_Gram_Kx = kernel.compute_kernelised_gram_matrix(data_x)
    mat_kernel_Gram_Ky = kernel.compute_kernelised_gram_matrix(data_y)
    mat_Kxy = kernel.compute_rectangular_kernel_matrix(data_x, data_y)

    if nx == ny:

        # TODO should I really subtract the diagonal for K_xy ???
        unbiased_mmd = (
            mat_kernel_Gram_Kx.sum() - np.diagonal(mat_kernel_Gram_Kx).sum() +
            mat_kernel_Gram_Ky.sum() - np.diagonal(mat_kernel_Gram_Ky).sum() -
            2 * (mat_Kxy.sum() - np.diagonal(mat_Kxy).sum())
        )
        unbiased_mmd /= (nx * (nx - 1))

    else:

        unbiased_mmd = (
            (1 / (nx * (nx - 1))) * (mat_kernel_Gram_Kx.sum() -
                                     np.diagonal(mat_kernel_Gram_Kx).sum()) +
            (1 / (ny * (ny - 1))) * (mat_kernel_Gram_Ky.sum() -
                                     np.diagonal(mat_kernel_Gram_Ky).sum()) -
            (2 / (nx * ny)) * mat_Kxy.sum()
        )

    dic = dict()
    dic['MMD'] = unbiased_mmd
    dic[_field_kernel_Gram_Kx] = mat_kernel_Gram_Kx
    dic[_field_kernel_Gram_Ky] = mat_kernel_Gram_Ky
    dic[_field_mat_Kxy] = mat_Kxy

    return dic


def compute_biased_squared_mmd(data_x, data_y, kernel):
    """
    Computes the biased estimate of the squared Maximum Mean Discrepancy
    between :math:`\text{P}_{\text{X}}` and :math:`\text{P}_{\text{Y}}`.

    Parameters
    ----------
    data_x : array_like
        The :math:`x_i`'s.
    data_y : array_like
        The :math:`y_i`'s.
    kernel : KernelWrapper
        The reproducing kernel associated to the RKHS chosen on the space.

    Returns
    -------
    dic
        A dictionary containing the value of :math:`\text{MMD}_{\text{b}}^2`,
        as well as kernel Gram matrix :math:`\text{K}_{\text{X}}`, kernel Gram
        matrix :math:`\text{K}_{\text{Y}}` and matrix
        :math:`\text{K}_{\text{XY}}`.
    """
    nx = data_x.shape[0]
    ny = data_y.shape[0]

    mat_kernel_Gram_Kx = kernel.compute_kernelised_gram_matrix(data_x)
    mat_kernel_Gram_Ky = kernel.compute_kernelised_gram_matrix(data_y)
    mat_Kxy = kernel.compute_rectangular_kernel_matrix(data_x, data_y)

    biased_mmd = (
            mat_kernel_Gram_Kx.sum() / (nx * nx) +
            mat_kernel_Gram_Ky.sum() / (ny * ny) -
            2 * mat_Kxy.sum() / (nx * ny)
    )

    dic = dict()
    dic['MMD'] = biased_mmd
    dic[_field_kernel_Gram_Kx] = mat_kernel_Gram_Kx
    dic[_field_kernel_Gram_Ky] = mat_kernel_Gram_Ky
    dic[_field_mat_Kxy] = mat_Kxy

    return dic


def perform_permutation_mmd(data_x, data_y, kernel, test_level,
                            nb_permutations=5000):

    mmd_dic = compute_biased_squared_mmd(
        data_x=data_x,
        data_y=data_y,
        kernel=kernel
    )
    mmd = mmd_dic['MMD']
    mat_kernel_Gram_Kx = mmd_dic[_field_kernel_Gram_Kx]
    mat_kernel_Gram_Ky = mmd_dic[_field_kernel_Gram_Ky]
    mat_Kxy = mmd_dic[_field_mat_Kxy]

    nx = mat_kernel_Gram_Kx.shape[0]
    ny = mat_kernel_Gram_Ky.shape[0]

    if nx != ny:

        msg = 'This method is designed for equal sample sizes for X and Y.'
        raise CaseNotHandled(msg)

    mat_kernel_Gram_Kxy = np.zeros((2 * nx, 2 * nx))
    mat_kernel_Gram_Kxy[0:nx, 0:nx] = mat_kernel_Gram_Kx
    mat_kernel_Gram_Kxy[nx:(2 * nx), nx:(2 * nx)] = mat_kernel_Gram_Ky
    mat_kernel_Gram_Kxy[0:nx, nx:(2 * nx)] = mat_Kxy
    mat_kernel_Gram_Kxy[nx:(2 * nx), 0:nx] = mat_Kxy.transpose()

    draws_from_null = np.zeros(nb_permutations)
    permuted_matrix = copy.deepcopy(mat_kernel_Gram_Kxy)
    for i in range(nb_permutations):
        np.random.shuffle(permuted_matrix)
        draws_from_null[i] = (
            np.sum(permuted_matrix[0:nx, 0:nx]) +
            np.sum(permuted_matrix[nx:(2 * nx), nx:(2 * nx)]) -
            np.sum(permuted_matrix[0:nx, nx:(2 * nx)]) -
            np.sum(permuted_matrix[nx:(2 * nx), 0:nx])
        ) / (nx * nx)

    # Compute the empirical quantile
    threshold = np.quantile(draws_from_null, 1 - test_level)

    dic = dict()
    dic['Reject H0 (H0 : P_X = P_Y)'] = mmd > threshold
    dic['MMD'] = mmd
    dic['Rejection threshold'] = threshold
    dic['Samples from simulated null'] = draws_from_null

    return dic


def perform_gram_matrix_spectrum_mmd(data_x, data_y, kernel, test_level,
                                     nb_simulations=5000):

    mmd_dic = compute_biased_squared_mmd(
        data_x=data_x,
        data_y=data_y,
        kernel=kernel
    )
    mmd = mmd_dic['MMD']
    mat_kernel_Gram_Kx = mmd_dic[_field_kernel_Gram_Kx]
    mat_kernel_Gram_Ky = mmd_dic[_field_kernel_Gram_Ky]
    mat_Kxy = mmd_dic[_field_mat_Kxy]

    nx = mat_kernel_Gram_Kx.shape[0]
    ny = mat_kernel_Gram_Ky.shape[0]

    if nx != ny:

        msg = 'This method is designed for equal sample sizes for X and Y.'
        raise CaseNotHandled(msg)

    mat_kernel_Gram_Kxy = np.zeros((2 * nx, 2 * nx))
    mat_kernel_Gram_Kxy[0:nx, 0:nx] = mat_kernel_Gram_Kx
    mat_kernel_Gram_Kxy[nx:(2 * nx), nx:(2 * nx)] = mat_kernel_Gram_Ky
    mat_kernel_Gram_Kxy[0:nx, nx:(2 * nx)] = mat_Kxy
    mat_kernel_Gram_Kxy[nx:(2 * nx), 0:nx] = mat_Kxy.transpose()

    mat_H = np.identity(2 * nx) - (1 / (2 * nx)) * np.ones((2 * nx, 2 * nx))
    mat_centered_kernel_Gram_Kxy = mat_H @ mat_kernel_Gram_Kxy @ mat_H

    # Simulating draws from the limit distribution
    eigenvalues = np.linalg.eigvalsh(mat_centered_kernel_Gram_Kxy)
    eigenvalues /= (2 * nx)
    nb_eigenvalues = eigenvalues.shape[0]
    draws_from_null = np.zeros((nb_simulations, 1))
    for i in range(nb_simulations):
        gaussians = sqrt(2) * norm.rvs(loc=0, scale=1, size=nb_eigenvalues)
        draws_from_null[i, 0] = np.dot(eigenvalues, gaussians ** 2)

    # Compute the empirical quantile
    threshold = np.quantile(draws_from_null, 1 - test_level)

    dic = dict()
    dic['Reject H0 (H0 : P_X = P_Y)'] = nx * mmd > threshold
    dic['MMD'] = nx * mmd
    dic['Rejection threshold'] = threshold
    dic['Samples from simulated null'] = draws_from_null

    return dic


def perform_mmd_test(data_x, data_y, kernel, test_level, scheme):

    if scheme is ImplementedMMDSchemes.SPECTRAL:

        test_dic = perform_gram_matrix_spectrum_mmd(
            data_x=data_x,
            data_y=data_y,
            kernel=kernel,
            test_level=test_level
        )

        return test_dic

    # elif scheme is ImplementedMMDSchemes.PERMUTATION:
    #
    #     test_dic = perform_permutation_mmd(
    #         data_x=data_x,
    #         data_y=data_y,
    #         kernel=kernel,
    #         test_level=test_level
    #     )
    #
    #     return test_dic

    else:

        msg = f"Scheme '{scheme}' not implemented for the MMD two-sample test."
        raise MMDTestingSchemeNotImplemented(msg)
