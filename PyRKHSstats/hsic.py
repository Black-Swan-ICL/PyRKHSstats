# TODO reformat docstrings
# This module contains the code to compute the HSIC and related quantities,
# using the tools developed in 'A Kernel Statistical Test of Independence', A.
# Gretton, K. Fukumizu, C. Hui Teo, L. Song, B. Scholkopf & A. J. Smola
# (NIPS # 21, 2007) which will be referred to as 'the paper' in the module.
import numpy as np

from PyRKHSstats.combinatorics_utilities import n_permute_m, \
    ordered_combinations

from PyRKHSstats.kernel_wrapper import KernelWrapper


def compute_squared_norm_mean_embedding(data, kernel):
    """
    Computes the estimate of the squared norm of the kernel mean embedding of
    the distribution generating the data.

    Parameters
    ----------
    data : array_like
        The observations in the domain space.
    kernel : CustomKernel
        The reproducing kernel associated to the RKHS on the domain space.

    Returns
    -------
    float
        The empirical estimate of the kernel mean embedding of the distribution
        generating the data.
    """
    m = data.shape[0]

    # TODO could be made simpler : 1_{1, m} x (K - diag(K)) x 1_{m, 1} is the
    #  same ; plus might avoid having to recalculate most of K !
    res = (1 / n_permute_m(m, 2)) * np.sum(
        [kernel.evaluate(data[i], data[j]) for i, j in
         ordered_combinations(m, n=2)]
    )

    return res


def biased_hsic(data_x, data_y, kernel_k, kernel_l):
    """
    Computes the biased empirical estimate of the Hilbert-Schmidt Independence
    Criterion (HSIC), as defined by equation (4) in the paper. Returns it along
    with matrices used in the computation (as those are expensive to compute,
    and may be needed when HSIC is needed).

    Parameters
    ----------
    data_x : array_like
        The observations in X space.
    data_y : array_like
        The observations in Y space.
    kernel_k : KernelWrapper
        The reproducing kernel associated to the RKHS on domain X.
    kernel_l : KernelWrapper
        The reproducing kernel associated to the RKHS on domain Y.

    Returns
    -------
    dic
        A dictionary containing the biased empirical estimate of the hsic, the
        k-kernelised Gram matrix, the l-kernelised Gram matrix and the centering
        matrix H.
    """
    m = data_x.shape[0]

    mat_K = kernel_k.compute_kernelised_gram_matrix(data_x)
    mat_L = kernel_l.compute_kernelised_gram_matrix(data_y)
    mat_H = np.identity(m) - (1/m) * np.ones((m, m))

    hsic = (1/m)**2 * np.trace(mat_K @ mat_H @ mat_L @ mat_H)

    dic = dict()
    dic['HSIC'] = hsic
    dic['K'] = mat_K
    dic['L'] = mat_L
    dic['H'] = mat_H

    return dic


def compute_estimate_biased_hsic_mean(data_x, data_y, kernel_k, kernel_l):
    """
    Computes an empirical estimate of the expectation of the Hilbert-Schmidt
    Independence Criterion under H_0.

    Parameters
    ----------
    data_x : array_like
        The observations in X space.
    data_y : array_like
        The observations in Y space.
    kernel_k : KernelWrapper
        The reproducing kernel associated to the RKHS on domain X.
    kernel_l : KernelWrapper
        The reproducing kernel associated to the RKHS on domain Y.

    Returns
    -------
    float
        An empirical estimate of the expectation of HSIC under H_0.
    """
    m = data_x.shape[0]

    squared_norm_mu_x = compute_squared_norm_mean_embedding(data=data_x,
                                                            kernel=kernel_k)
    squared_norm_mu_y = compute_squared_norm_mean_embedding(data=data_y,
                                                            kernel=kernel_l)
    biased_hsic_mean = (1 / m) * (1 + squared_norm_mu_x * squared_norm_mu_y -
                                  squared_norm_mu_x - squared_norm_mu_y)

    return biased_hsic_mean


def compute_estimate_biased_hsic_variance(mat_K, mat_L, mat_H):
    """
    Computes an empirical estimate of the variance of the Hilbert-Schmidt
    Independence Criterion under H_0.

    Parameters
    ----------
    mat_K : array_like
        The kernelised Gram matrix (kernel_k(data_x[i], data_x[j]))_{i, j}.
    mat_L : array_like
        The kernelised Gram matrix (kernel_l(data_y[i], data_y[j]))_{i, j}.
    mat_H : array_like
        The centering matrix.

    Returns
    -------
    float
        An empirical estimate of the variance of HSIC under H_0.
    """
    m = mat_K.shape[0]

    mat_B = np.power(
        np.multiply(mat_H @ mat_K @ mat_H, mat_H @ mat_L @ mat_H),
        2
    )
    vec_ones = np.ones((m, 1))

    biased_hsic_variance = (
            ((2 * (m - 4) * (m - 5)) / (n_permute_m(m, 4) * m * (m - 1))) *
            vec_ones.T @ (mat_B - np.diagflat(np.diag(mat_B))) @ vec_ones
    ).flatten()[0]

    return biased_hsic_variance
