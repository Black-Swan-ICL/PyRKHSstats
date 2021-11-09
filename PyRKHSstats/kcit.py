"""
This module contains the code to conduct the Kernel-based Conditional
Independence Test as defined in 'Kernel-based Conditional Independence Test and
Application in Causal Discovery', K. Zhang, J. Peters, D. Janzing and B.
Sch\"{o}lkopf (UAI #27, 2011) which will be referred to as 'the paper' in the
module.
"""
import numpy as np

from scipy.stats import gamma


def compute_tci(data_x, data_y, data_z, kernel_kx, kernel_ky, kernel_kz,
                epsilon):
    """
    Computes :math:`\widetilde{\text{T}}_{\text{CI}}` the test statistic for
    KCIT, as presented in the paper. Also provides some matrices used in the
    computation that are expensive to compute and re-used in other parts of
    KCIT.

    Parameters
    ----------
    data_x : array_like
        The observations in :math:`\mathcal{X}` space.
    data_y : array_like
        The observations in :math:`\mathcal{Y}` space.
    data_z : array_like
        The observations in :math:`\mathcal{Z}` space.
    kernel_kx : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{X}`.
    kernel_ky : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{Y}`.
    kernel_kz : KernelWrapper
        The reproducing kernel associated to the RKHS on domain
        :math:`\mathcal{Z}`.
    epsilon : float
        A regularisation parameter used in the computation of matrix
        :math:`\text{R}_{\text{Z}}` in the paper.

    Returns
    -------
    dict
        A dictionary containing the value of
        :math:`\widetilde{\text{T}}_{\text{CI}}` and matrices
        :math:`\widetilde{\text{K}}_{\ddot{\text{X}} | \text{Z}}` and
        :math:`\widetilde{\text{K}}_{\text{Y} | \text{Z}}`.
    """

    n = data_x.shape[0]

    mat_H = np.identity(n) - (1/n) * np.ones((n, n))

    mat_Kx = kernel_kx.compute_kernelised_gram_matrix(data_x)
    mat_Ky = kernel_ky.compute_kernelised_gram_matrix(data_y)
    mat_Kz = kernel_kz.compute_kernelised_gram_matrix(data_z)

    # Computing $\text{R}_{\text{Z}}$
    mat_tilde_Kz = mat_H @ mat_Kz @ mat_H
    mat_Rz = epsilon * np.linalg.inv(mat_tilde_Kz + epsilon * np.identity(n))

    # Computing $\widetilde{\text{K}}_{\ddot{\text{X}}}$
    mat_Kddotx = np.multiply(mat_Kx, mat_Kz)
    mat_tilde_Kddotx = mat_H @ mat_Kddotx @ mat_H
    # Computing $\widetilde{\text{K}}_{\ddot{\text{X}} | \text{Z}}$
    mat_tilde_Kddotx_given_z = mat_Rz @ mat_tilde_Kddotx @ mat_Rz

    # Computing $\widetilde{\text{K}}_{\text{Y}}$
    mat_tilde_Ky = mat_H @ mat_Ky @ mat_H
    # Computing $\widetilde{\text{K}}_{\text{Y} | \text{Z}}$
    mat_tilde_Ky_given_z = mat_Rz @ mat_tilde_Ky @ mat_Rz

    tci = (1 / n) * np.trace(mat_tilde_Kddotx_given_z @ mat_tilde_Ky_given_z)

    dic = dict()
    dic['TCI'] = tci
    dic['tildeK_ddotX_given_Z'] = mat_tilde_Kddotx_given_z
    dic['tildeK_Y_given_Z'] = mat_tilde_Ky_given_z

    return dic


def compute_mat_tilde_W(mat_tilde_Kddotx_given_z, mat_tilde_Ky_given_z):
    """
    Computes :math:`\widetilde{W}` as defined in the paper ; it is used in
    the Gamma approximation scheme for KCIT.

    Parameters
    ----------
    mat_tilde_Kddotx_given_z : array_like
        Matrix :math:`\widetilde{\text{K}}_{\ddot{\text{X}} | \text{Z}}` as
        defined in the paper.
    mat_tilde_Ky_given_z : array_like
        Matrix :math:`\widetilde{\text{K}}_{\text{Y} | \text{Z}}` as defined in
        the paper.

    Returns
    -------
    array_like
        Matrix :math:`\widetilde{W}` as defined in the paper.
    """

    n = mat_tilde_Kddotx_given_z.shape[0]

    # Computing the EVD of $\widetilde{\text{K}}_{\ddot{\text{X}} | \text{Z}}$
    eigenval_mat_tilde_Kddotx_given_z, eigenvec_mat_tilde_Kddotx_given_z = (
        np.linalg.eig(mat_tilde_Kddotx_given_z)
    )
    # The eigenvalues need to be arranged in descending order
    descending_indices = eigenval_mat_tilde_Kddotx_given_z.argsort()[::-1]
    eigenval_mat_tilde_Kddotx_given_z = np.diag(
        eigenval_mat_tilde_Kddotx_given_z[descending_indices]
    )
    eigenvec_mat_tilde_Kddotx_given_z = eigenvec_mat_tilde_Kddotx_given_z[
        :, descending_indices
    ]
    mat_psi = eigenvec_mat_tilde_Kddotx_given_z @ np.sqrt(
        eigenval_mat_tilde_Kddotx_given_z
    )

    # Computing the EVD of $\widetilde{\text{K}}_{\text{Y} | \text{Z}}$
    eigenval_mat_tilde_Ky_given_z, eigenvec_mat_tilde_Ky_given_z = (
        np.linalg.eig(mat_tilde_Ky_given_z)
    )
    # The eigenvalues need to be arranged in descending order
    descending_indices = eigenval_mat_tilde_Ky_given_z.argsort()[::-1]
    eigenval_mat_tilde_Ky_given_z = np.diag(
        eigenval_mat_tilde_Ky_given_z[descending_indices]
    )
    eigenvec_mat_tilde_Ky_given_z = eigenvec_mat_tilde_Ky_given_z[
        :, descending_indices
    ]
    mat_phi = eigenvec_mat_tilde_Ky_given_z @ np.sqrt(
        eigenval_mat_tilde_Ky_given_z
    )

    # Computing matrix $\widetilde{W}$
    mat_tilde_W = np.zeros((n * n, n))
    for i in range(n):
        vec_tilde_w = np.outer(mat_psi[i, :], mat_phi[i, :])
        vec_tilde_w = vec_tilde_w.flatten('C')
        mat_tilde_W[:, i] = vec_tilde_w

    return np.ones((n, n))


def calibrate_tci_gamma_approximation(mat_tilde_Kddotx_given_z,
                                      mat_tilde_Ky_given_z):
    """
    Returns a calibrated, frozen Gamma distribution, ready for use for a
    Gamma approximation KCIT test.

    Parameters
    ----------
    mat_tilde_Kddotx_given_z : array_like
        Matrix :math:`\widetilde{\text{K}}_{\ddot{\text{X}} | \text{Z}}` as
        defined in the paper.
    mat_tilde_Ky_given_z : array_like
        Matrix :math:`\widetilde{\text{K}}_{\text{Y} | \text{Z}}` as defined in
        the paper.

    Returns
    -------
    scipy.stats._distn_infrastructure.rv_frozen
        The Gamma distribution, calibrated and ready to use for the so-called
        Gamma approximation in KCIT conditional independence testing.
    """

    n = mat_tilde_Kddotx_given_z.shape[0]
    mat_tilde_W = compute_mat_tilde_W(
        mat_tilde_Kddotx_given_z=mat_tilde_Kddotx_given_z,
        mat_tilde_Ky_given_z=mat_tilde_Ky_given_z
    )

    mat_tilde_W_tilde_W_transpose = mat_tilde_W @ mat_tilde_W.transpose()
    mean_tci = (1 / n) * np.trace(mat_tilde_W_tilde_W_transpose)
    var_tci = (2 / (n * n)) * np.trace(mat_tilde_W_tilde_W_transpose @
                                       mat_tilde_W_tilde_W_transpose)

    alpha = mean_tci ** 2 / var_tci
    beta = var_tci / mean_tci

    return gamma(alpha, loc=0, scale=beta)
