# TODO make into a class to make easy to persist (text output, bespoke
#  serialisation etc)
# TODO add computation time ?
# TODO reformat docstrings
from PyRKHSstats.gamma_approximation import calibrate_gamma


def hsic_independence_test(data_x, data_y, kernel_k, kernel_l, hsic_func,
                           test_level):
    """

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
    dic = hsic_func(data_x, data_y, kernel_k, kernel_l)
    hsic = dic['HSIC']
    mat_K = dic['K']
    mat_L = dic['L']
    mat_H = dic['H']
    m = mat_K.shape[0]

    # Calibrate the Gamma approximation to the distribution of HSIC under H_0
    gamma_distribution = calibrate_gamma(data_x=data_x,
                                         data_y=data_y,
                                         kernel_k=kernel_k,
                                         kernel_l=kernel_l,
                                         mat_K=mat_K,
                                         mat_L=mat_L,
                                         mat_H=mat_H)
    threshold = gamma_distribution.ppf(1 - test_level).flatten()[0]

    dic = dict()
    dic['RejectH0'] = (m * hsic) > threshold
    dic['HSIC'] = hsic
    dic['m'] = m
    dic['m * HSIC'] = m * hsic
    dic['Threshold'] = threshold
    dic['Gamma'] = gamma_distribution

    return dic
