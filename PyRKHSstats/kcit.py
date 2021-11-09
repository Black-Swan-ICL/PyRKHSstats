"""
This module contains the code to conduct the Kernel-based Conditional
Independence Test as defined in 'Kernel-based Conditional Independence Test and
Application in Causal Discovery', K. Zhang, J. Peters, D. Janzing and B.
Sch\"{o}lkopf (UAI #27, 2011).
"""
import numpy as np

from scipy.stats import gamma


def compute_tci(data_x, data_y, data_z, kernel_kx, kernel_ky, kernel_kz,
                epsilon):

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

    n = mat_tilde_Kddotx_given_z.shape[0]

    # Computing the EVD of $\widetilde{\text{K}}_{\ddot{\text{X}} | \text{Z}}$
    eigenval_mat_tilde_Kddotx_given_z, eigenvec_mat_tilde_Kddotx_given_z = (
        np.linalg.eig(mat_tilde_Kddotx_given_z)
    )
    mat_phi = eigenvec_mat_tilde_Kddotx_given_z @ np.sqrt(
        eigenval_mat_tilde_Kddotx_given_z
    )

    # Computing the EVD of $\widetilde{\text{K}}_{\text{Y} | \text{Z}}$
    eigenval_mat_tilde_Ky_given_z, eigenvec_mat_tilde_Ky_given_z = (
        np.linalg.eig(mat_tilde_Ky_given_z)
    )
    mat_psi = eigenvec_mat_tilde_Ky_given_z @ np.sqrt(
        eigenval_mat_tilde_Ky_given_z
    )

    return np.ones((n, n))


def calibrate_tci_gamma_approximation(mat_tilde_Kddotx_given_z,
                                      mat_tilde_Ky_given_z):

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


