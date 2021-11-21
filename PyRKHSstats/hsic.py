"""
This module contains the code to compute the HSIC and related quantities,
using the tools developed in 'A Kernel Statistical Test of Independence', A.
Gretton, K. Fukumizu, C. Hui Teo, L. Song, B. Scholkopf & A. J. Smola
(NIPS # 21, 2007) which will be referred to as 'the paper' in the module.
"""
import numpy as np

from enum import Enum, unique

from scipy.stats import gamma

from PyRKHSstats.combinatorics_utilities import n_permute_m, \
    ordered_combinations, generate_strict_permutations
from PyRKHSstats.kernel_wrapper import KernelWrapper


@unique
class ImplementedHSICScheme(Enum):
    GAMMA = 'Gamma Approximation'
    PERMUTATION = 'Permutation of Samples'


class HSICTestingSchemeNotImplemented(Exception):
    """Raised when the user requests a scheme for HSIC that is not implemented.
    """


def compute_squared_norm_mean_embedding(data, kernel):
    """
    Computes the estimate of the squared norm of the kernel mean embedding of
    the distribution generating the data.

    Parameters
    ----------
    data : array_like
        The observations in the domain space.
    kernel : KernelWrapper
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


def compute_biased_hsic(data_x, data_y, kernel_kx, kernel_ky):
    """
    Computes the biased empirical estimate of the Hilbert-Schmidt Independence
    Criterion (HSIC), as defined by equation (4) in the paper. Returns it along
    with matrices used in the computation (as those are expensive to compute,
    and may be needed when HSIC is needed).

    Parameters
    ----------
    data_x : array_like
        The observations in :math:`\mathcal{X}` space.
    data_y : array_like
        The observations in :math:`\mathcal{Y}` space.
    kernel_kx : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{X}`.
    kernel_ky : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{Y}`.

    Returns
    -------
    dic
        A dictionary containing the biased empirical estimate of HSIC, the
        kernel Gram matrices :math:`\text{K}_{\text{X}}` and
        :math:`\text{K}_{\text{Y}}` and the centering matrix :math:`\text{H}`.
    """
    m = data_x.shape[0]

    mat_K = kernel_kx.compute_kernelised_gram_matrix(data_x)
    mat_L = kernel_ky.compute_kernelised_gram_matrix(data_y)
    mat_H = np.identity(m) - (1/m) * np.ones((m, m))

    hsic = (1/m)**2 * np.trace(mat_K @ mat_H @ mat_L @ mat_H)

    dic = dict()
    dic['HSIC'] = hsic
    dic['K'] = mat_K
    dic['L'] = mat_L
    dic['H'] = mat_H

    return dic


def compute_estimate_biased_hsic_mean(data_x, data_y, kernel_kx, kernel_ky):
    """
    Computes an empirical estimate of the expectation of the Hilbert-Schmidt
    Independence Criterion under :math:`\text{H}_0`.

    Parameters
    ----------
    data_x : array_like
        The observations in :math:`\mathcal{X}` space.
    data_y : array_like
        The observations in :math:`\mathcal{Y}` space.
    kernel_kx : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{X}`.
    kernel_ky : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{Y}`.

    Returns
    -------
    float
        An empirical estimate of the expectation of HSIC under
        :math:`\text{H}_0`.
    """
    m = data_x.shape[0]

    squared_norm_mu_x = compute_squared_norm_mean_embedding(data=data_x,
                                                            kernel=kernel_kx)
    squared_norm_mu_y = compute_squared_norm_mean_embedding(data=data_y,
                                                            kernel=kernel_ky)
    biased_hsic_mean = (1 / m) * (1 + squared_norm_mu_x * squared_norm_mu_y -
                                  squared_norm_mu_x - squared_norm_mu_y)

    return biased_hsic_mean


def compute_estimate_biased_hsic_variance(mat_K, mat_L, mat_H):
    """
    Computes an empirical estimate of the variance of the Hilbert-Schmidt
    Independence Criterion under :math:`\text{H}_0`.

    Parameters
    ----------
    mat_K : array_like
        The kernel Gram matrix :math:`\text{K}_{\text{X}}`.
    mat_L : array_like
        The kernel Gram matrix :math:`\text{K}_{\text{Y}}`.
    mat_H : array_like
        The centering matrix.

    Returns
    -------
    float
        An empirical estimate of the variance of HSIC under
        :math:`\text{H}_0`.
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


def calibrate_hsic_gamma_approximation(data_x, data_y, kernel_kx, kernel_ky,
                                       mat_K, mat_L, mat_H):
    """
    Returns a calibrated, frozen Gamma distribution, ready for use for Gamma
    approximation in HSIC independence testing.

    Parameters
    ----------
    data_x : array_like
        The observations in :math:`\mathcal{X}` space.
    data_y : array_like
        The observations in :math:`\mathcal{Y}` space.
    kernel_kx : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{X}`.
    kernel_ky : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{Y}`.
    mat_K : array_like
        The kernel Gram matrix :math:`\text{K}_{\text{X}}`.
    mat_L : array_like
        The kernel Gram matrix :math:`\text{K}_{\text{Y}}`.
    mat_H : array_like
        The centering matrix.

    Returns
    -------
    scipy.stats._distn_infrastructure.rv_frozen
        The Gamma distribution, calibrated and ready to use for the so-called
        Gamma approximation in HSIC independence testing.
    """
    biased_hsic_mean = compute_estimate_biased_hsic_mean(data_x=data_x,
                                                         data_y=data_y,
                                                         kernel_kx=kernel_kx,
                                                         kernel_ky=kernel_ky)
    biased_hsic_variance = compute_estimate_biased_hsic_variance(mat_K=mat_K,
                                                                 mat_L=mat_L,
                                                                 mat_H=mat_H)
    m = data_x.shape[0]
    alpha = biased_hsic_mean**2 / biased_hsic_variance
    beta = m * biased_hsic_variance / biased_hsic_mean

    return gamma(alpha, loc=0, scale=beta)


def perform_gamma_approximation_hsic_independence_testing(
        data_x, data_y, kernel_kx, kernel_ky, hsic_func=compute_biased_hsic,
        test_level=0.01):
    """
    Performs the HSIC unconditional independence test using the Gamma
    approximation scheme.

    Parameters
    ----------
    data_x : array_like
        The observations in :math:`\mathcal{X}` space.
    data_y : array_like
        The observations in :math:`\mathcal{Y}` space.
    kernel_kx : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{X}`.
    kernel_ky : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{Y}`.
    hsic_func : callable
        The function used to compute the HSIC.
    test_level : float
        The upper bound on the probability of error or type I we may accept.

    Returns
    -------
    dic
        A dictionary containing the value of HSIC, the threshold for acceptance
        of the test, a boolean indicating whether to reject the null
        hypothesis of independence of x and y as well as the calibrated
        Gamma distribution used.
    """
    dic = hsic_func(data_x, data_y, kernel_kx, kernel_ky)
    hsic = dic['HSIC']
    mat_K = dic['K']
    mat_L = dic['L']
    mat_H = dic['H']
    m = mat_K.shape[0]

    # Calibrate the Gamma approximation to the distribution of HSIC under H_0
    calibrated_gamma = calibrate_hsic_gamma_approximation(
        data_x=data_x,
        data_y=data_y,
        kernel_kx=kernel_kx,
        kernel_ky=kernel_ky,
        mat_K=mat_K,
        mat_L=mat_L,
        mat_H=mat_H
    )
    threshold = calibrated_gamma.ppf(1 - test_level).flatten()[0]

    dic = dict()
    dic['Reject H0 (H0 : X _||_ Y | Z)'] = m * hsic > threshold
    dic['HSIC'] = m * hsic  # NOT an error : it is m HSIC that follows a Gamma
    dic['Rejection threshold'] = threshold
    dic['Gamma distribution'] = calibrated_gamma

    return dic


def perform_permutation_hsic_independence_testing(
        data_x, data_y, kernel_kx, kernel_ky, permutations=None,
        hsic_func=compute_biased_hsic, test_level=0.01):
    """
    Performs the HSIC unconditional independence test using the resampling
    scheme (permuting the observations in :math:`\mathcal{X}` space while
    leaving the observations in :math:`\mathcal{Y}` space unchanged).

    Parameters
    ----------
    data_x : array_like
        The observations in :math:`\mathcal{X}` space.
    data_y : array_like
        The observations in :math:`\mathcal{Y}` space.
    kernel_kx : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{X}`.
    kernel_ky : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{Y}`.
    permutations : set
        A set of permutations to use for the resampling.
    hsic_func : callable
        The function used to compute the HSIC.
    test_level : float
        The upper bound on the probability of error or type I we may accept.
    Returns
    -------
    dic
        A dictionary containing the value of HSIC, the threshold for acceptance
        of the test, and a boolean indicating whether to reject the null
        hypothesis of independence of x and y.
    """

    nb_obs, nb_dim = data_x.shape

    dic = hsic_func(data_x, data_y, kernel_kx, kernel_ky)
    hsic = dic['HSIC']

    # Simulate the limit distribution by resampling
    if permutations is None:
        permutations = generate_strict_permutations(
            indices_to_permute=list(range(nb_obs)),
            nb_permutations_wanted=5000
        )

    hsic_samples = np.zeros(len(permutations))
    for iter, permutation in enumerate(permutations):

        permuted_data_x = np.zeros_like(data_x)
        for j in range(nb_dim):
            permuted_data_x[:, j] = data_x[permutation, j]

        hsic_samples[iter] = hsic_func(
            data_x=permuted_data_x,
            data_y=data_y,
            kernel_kx=kernel_kx,
            kernel_ky=kernel_ky
        )['HSIC']

    threshold = np.quantile(hsic_samples, 1 - test_level)

    dic = dict()
    dic['Reject H0 (H0 : X _||_ Y | Z)'] = hsic > threshold
    dic['HSIC'] = hsic  # NOT an error : here we work with HSIC itself
    dic['Rejection threshold'] = threshold

    return dic


def perform_hsic_independence_testing(data_x, data_y, kernel_kx, kernel_ky,
                                      test_level, scheme):
    """
    Performs the HSIC unconditional independence test using the scheme
    specified by the user.

    Parameters
    ----------
    data_x : array_like
        The observations in :math:`\mathcal{X}` space.
    data_y : array_like
        The observations in :math:`\mathcal{Y}` space.
    kernel_kx : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{X}`.
    kernel_ky : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{Y}`.
    test_level : float
        The level of the test.
    scheme : ImplementedHSICScheme
        The testing scheme to use.

    Returns
    -------
    dict
        A dictionary containing the result of the test and additional
        information.
    """

    if scheme is ImplementedHSICScheme.GAMMA:

        test_dic = perform_gamma_approximation_hsic_independence_testing(
            data_x=data_x,
            data_y=data_y,
            kernel_kx=kernel_kx,
            kernel_ky=kernel_ky,
            test_level=test_level
        )

        return test_dic

    elif scheme is ImplementedHSICScheme.PERMUTATION:

        test_dic = perform_permutation_hsic_independence_testing(
            data_x=data_x,
            data_y=data_y,
            kernel_kx=kernel_kx,
            kernel_ky=kernel_ky,
            test_level=test_level
        )

        return test_dic

    else:

        msg = (
            f"Scheme '{scheme}' not implemented for HSIC-based " +
            "independence testing."
        )
        raise HSICTestingSchemeNotImplemented(msg)
