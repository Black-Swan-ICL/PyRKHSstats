"""
This module implements the HSCIC and auxiliary quantities, as introduced in
'A Measure-Theoretic Approach to Kernel Conditional Mean Embedding' (J. Park,
and K. Muandet, 2020, NeurIPS #33) which will be referred to as 'the paper' in
this module. Numpy is used.
"""
import math
import numpy as np


def compute_mat_W(mat_K_Z, regularisation_constant):
    """
    Computes the matrix :math:`W`, as specified in the paper.

    Matrix :math:`W` is defined as follows :

    .. math::
        W := (K_Z + n \lambda I_n)^{-1}

    Parameters
    ----------
    mat_K_Z : array_like
        The kernel Gram matrix for the :math:`Z` variable :math:`K_Z`.
    regularisation_constant : float
        The regularisation constant.

    Returns
    -------
    array_like
        The matrix :math:`W` as defined in the paper.
    """
    n = mat_K_Z.shape[0]
    mat_W = np.linalg.inv(
        mat_K_Z + n * regularisation_constant * np.identity(n)
    )

    return mat_W


def compute_vec_k_Z_in_z(z, data_z, kernel_z):
    """
    Evaluates the function :math:`k_Z(z)`, as defined in the paper.

    Function :math:`k_Z(z)` is defined on :math:`\mathcal{Z}` by

    .. math::
        k_Z(z) = (k_Z(z_1, z), \dots , k_Z(z_n, z))^T

    Parameters
    ----------
    z : array_like
        The evaluation point :math:`z`.
    data_z : array_like
        The observations (data) in :math:`\mathcal{Z}` domain.
    kernel_z : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{Z}` .

    Returns
    -------
    array_like
        The vector valued function evaluated at :math:`z`, :math:`k_Z(z)`.
    """
    n = len(data_z)
    res = np.zeros((n, 1))
    for i in range(n):

        res[i, 0] = kernel_z.evaluate(data_z[i], z)

    return res


def compute_hscic(z, mat_K_X, mat_K_Y, hadamard_K_X_K_Y, mat_W, func_vec_k_Z):
    """
    Computes the empirical HSCIC, as defined in the paper.

    The empirical HSCIC is defined by

    .. math::
        \widehat{H}^2_{X, Y | Z}(z) = k_Z(z)^T W (K_X \odot K_Y) W^T k_Z(z) -
        2 k_Z(z)^T W ((K_X W^T k_Z(z)) \odot (K_Y W^T k_Z(z))) +
        (k_Z(z)^T W K_X W^T k_Z(z))(k_Z(z)^T W K_Y W^T k_Z(z)).

    Parameters
    ----------
    z : array_like
        The evaluation point :math:`z`.
    mat_K_X : array_like
        The kernel Gram matrix for the :math:`X` variable :math:`K_X`.
    mat_K_Y : array_like
        The kernel Gram matrix for the :math:`Y` variable :math:`K_Y`.
    hadamard_K_X_K_Y : array_like
        The Hadamard product of :math:`K_X` and :math:`K_Y`.
    mat_W : array_like
        The matrix :math:`W`, as defined in the paper.
    func_vec_k_Z : callable
        The function :math:`k_Z(z)`, as defined in the paper.

    Returns
    -------
    float
        The empirical HSCIC evaluated at :math:`z`.
    """
    k_Z_in_z = func_vec_k_Z(z)

    # To avoid repeated matrix computations
    k_Z_in_z_W = k_Z_in_z.T @ mat_W
    K_X_k_Z_in_z_W = mat_K_X @ k_Z_in_z_W.T
    K_Y_k_Z_in_z_W = mat_K_Y @ k_Z_in_z_W.T

    term_1 = k_Z_in_z_W @ hadamard_K_X_K_Y @ k_Z_in_z_W.T
    term_2 = k_Z_in_z_W @ np.multiply(K_X_k_Z_in_z_W, K_Y_k_Z_in_z_W)
    term_3 = (k_Z_in_z_W @ K_X_k_Z_in_z_W) * (k_Z_in_z_W @ K_Y_k_Z_in_z_W)

    res = math.sqrt(term_1 - 2 * term_2 + term_3)

    return res
