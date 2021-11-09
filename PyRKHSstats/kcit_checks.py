import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from scipy.stats import norm, gamma, laplace
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.kcit import compute_tci, calibrate_tci_gamma_approximation

if __name__ == '__main__':

    test_level = 0.01

    N = 100

    # Low-dimensional where CI holds true
    data_z = laplace.rvs(loc=0, scale=1, size=N).reshape(-1, 1)
    data_x = 2 * data_z + laplace.rvs(loc=0, scale=1, size=N).reshape(-1, 1)
    data_y = data_z + laplace.rvs(loc=0, scale=1, size=N).reshape(-1, 1)

    # Kernels to use
    length_scale_kx = np.median(np.abs(pdist(data_x)))
    kernel_kx = KernelWrapper(RBF(length_scale=length_scale_kx))
    length_scale_ky = np.median(np.abs(pdist(data_y)))
    kernel_ky = KernelWrapper(RBF(length_scale=length_scale_ky))
    length_scale_kz = np.median(np.abs(pdist(data_z)))
    kernel_kz = KernelWrapper(RBF(length_scale=length_scale_kz))

    # Hyperparameter - arbitrary value
    epsilon = 0.001

    kcit_dic = compute_tci(
        data_x=data_x,
        data_y=data_y,
        data_z=data_z,
        kernel_kx=kernel_kx,
        kernel_ky=kernel_ky,
        kernel_kz=kernel_kz,
        epsilon=epsilon
    )
    tci = kcit_dic['TCI']
    mat_tilde_Kddotx_given_z = kcit_dic['tildeK_ddotX_given_Z']
    mat_tilde_Ky_given_z = kcit_dic['tildeK_Y_given_Z']

    calibrated_gamma = calibrate_tci_gamma_approximation(
        mat_tilde_Kddotx_given_z=mat_tilde_Kddotx_given_z,
        mat_tilde_Ky_given_z=mat_tilde_Ky_given_z
    )

    threshold = calibrated_gamma.ppf(1 - test_level).flatten()[0]

    print(tci)
    print(threshold)
    print(tci > threshold)
