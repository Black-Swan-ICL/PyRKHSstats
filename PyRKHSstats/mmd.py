"""
This module contains the code to compute the Maximum Mean Discrepancy (MMD)
following the material presented in papers 'A Fast, Consistent Kernel
Two-Sample Test', A. Gretton, K. Fukumizu, Z. Harchaoui and B. K.
Sriperumbudur (NIPS #22, 2009) and 'A Kernel Two-Sample Test', A. Gretton,
K. M. Borgwardt, M. J. Rasch, B. Sch\"{o}lkopf and A. Smola (Journal of
Machine Learning Research #13, 2012).
"""
import numpy as np


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
    float
        The unbiased estimate of the squared MMD.

    Notes
    -----
    As pointed out by A. Gretton and co-authors in their 2012 'A Kernel
    Two-Sample Test' (paragraph following the proof of lemma 6), computing
    the unbiased estimator of the squared MMD can yield negative values.
    """

    nx = data_x.shape[0]
    ny = data_y.shape[0]

    mat_Kx = kernel.compute_kernelised_gram_matrix(data_x)
    mat_Ky = kernel.compute_kernelised_gram_matrix(data_y)
    mat_Kxy = kernel.compute_rectangular_kernel_matrix(data_x, data_y)

    if nx == ny:

        unbiased_mmd = (
                mat_Kx.sum() - np.diagonal(mat_Kx).sum() +
                mat_Ky.sum() - np.diagonal(mat_Ky).sum() -
                2 * (mat_Kxy.sum() - np.diagonal(mat_Kxy).sum())
        )
        unbiased_mmd /= (nx * (nx - 1))

        return unbiased_mmd

    else:

        unbiased_mmd = (
                (mat_Kx.sum() - np.diagonal(mat_Kx).sum()) / (nx * (nx - 1)) +
                (mat_Ky.sum() - np.diagonal(mat_Ky).sum()) / (ny * (ny - 1)) -
                2 * mat_Kxy.sum() / (nx * ny)
        )

        return unbiased_mmd


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
    float
        The biased estimate of the squared MMD.
    """
    nx = data_x.shape[0]
    ny = data_y.shape[0]

    mat_Kx = kernel.compute_kernelised_gram_matrix(data_x)
    mat_Ky = kernel.compute_kernelised_gram_matrix(data_y)
    mat_Kxy = kernel.compute_rectangular_kernel_matrix(data_x, data_y)

    biased_mmd = (
            mat_Kx.sum() / (nx * nx) +
            mat_Ky.sum() / (ny * ny) -
            2 * mat_Kxy.sum() / (nx * ny)
    )

    return biased_mmd
