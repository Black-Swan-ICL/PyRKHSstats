# TODO reformat docstrings
# Tools to perform the Gamma approximation for HSIC, as described in 'A Kernel
# Statistical Test of Independence', A. Gretton, K. Fukumizu, C. Hui Teo,
# L. Song, B. Scholkopf & A. J. Smola (NIPS # 21, 2007)

from scipy.stats import gamma

from PyRKHSstats.hsic import compute_estimate_biased_hsic_mean, \
    compute_estimate_biased_hsic_variance


def calibrate_gamma(data_x, data_y, kernel_k, kernel_l, mat_K, mat_L, mat_H):
    """
    Returns a calibrated, frozen Gamma distribution, ready for use for Gamma
    approximation in HSIC independence testing.

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
    mat_K : array_like
        The kernelised Gram matrix (kernel_k(data_x[i], data_x[j]))_{i, j}.
    mat_L : array_like
        The kernelised Gram matrix (kernel_l(data_y[i], data_y[j]))_{i, j}.
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
                                                         kernel_k=kernel_k,
                                                         kernel_l=kernel_l)
    biased_hsic_variance = compute_estimate_biased_hsic_variance(mat_K=mat_K,
                                                                 mat_L=mat_L,
                                                                 mat_H=mat_H)
    m = data_x.shape[0]
    alpha = biased_hsic_mean**2 / biased_hsic_variance
    beta = m * biased_hsic_variance / biased_hsic_mean

    return gamma(alpha, loc=0, scale=beta)
