# This module contains the code to contain the estimator for NFSIC, as developed
# in 'An Adaptive Test of Independence with Analytic Kernel Embeddings', W.
# Jitkrittum, Z. Szab\'{o} & A. Gretton (ICML # 34, 2017) which will be referred
# to as 'the paper' in the module.
# TODO debug - the limit distribution does not appear as a Chi2(J) !
import numpy as np


def normalised_fsic(data_x, data_y, test_points_v, test_points_w, kernel_k,
                    kernel_l, gamma):
    """
    Computes an estimate of the Normalised Finite-Set Independence Criterion
    (NFSIC), as defined in proposition 6 in the paper.

    Parameters
    ----------
    data_x : array_like
        The observations in X space.
    data_y : array_like
        The observations in Y space.
    test_points_v : array_like
        The test points in X space.
    test_points_w : array_like
        The test points in Y space.
    kernel_k : KernelWrapper
        The reproducing kernel associated to the RKHS on domain X.
    kernel_l : KernelWrapper
        The reproducing kernel associated to the RKHS on domain Y.
    gamma : float
        The regularisation parameter.

    Returns
    -------
    float
        The estimate of the NFSIC.
    """
    n = data_x.shape[0]
    J = test_points_v.shape[0]

    # Compute the cross (test points, observations) kernel matrices
    mat_K = kernel_k.compute_rectangular_kernel_matrix(test_points_v, data_x)
    mat_L = kernel_l.compute_rectangular_kernel_matrix(test_points_w, data_y)

    term_K_L_1 = np.multiply(mat_K, mat_L) @ np.ones((n, 1))
    term_K_1_L_1 = np.multiply(mat_K @ np.ones((n, 1)), mat_L @ np.ones((n, 1)))

    # Computing \hat{u}
    u_hat = (1 / (n - 1)) * term_K_L_1 - (1 / (n * (n - 1))) * term_K_1_L_1

    # Computing \widehat{\Sigma}
    biased_u_hat = (1 / n) * term_K_L_1 - (1 / (n * n)) * term_K_1_L_1
    mat_H = np.identity(n) - (1 / n) * np.ones((n, n))
    term_K_H = mat_K @ mat_H
    term_L_H = mat_L @ mat_H
    mat_gamma = np.multiply(term_K_H, term_L_H) - biased_u_hat @ np.ones((1, n))
    sigma_hat = (1 / n) * mat_gamma @ np.transpose(mat_gamma)

    reg_concentration_mat = np.linalg.inv(sigma_hat + gamma * np.identity(J))

    nfsic = n * np.transpose(u_hat) @ reg_concentration_mat @ u_hat

    return nfsic

