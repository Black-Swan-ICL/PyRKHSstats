import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, gamma, laplace
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.mmd import compute_unbiased_squared_mmd, compute_biased_mmd


if __name__ == '__main__':

    # We start by checking we can replicate the results presented in figure 2
    # in 'A Kernel Two-Sample Test', A. Gretton, K. M. Borgwardt, M. J. Rasch,
    # B. Sch\"{o}lkopf and A. Smola (Journal of Machine Learning Research
    # #13, 2012)
    savedir = os.path.join('checks', 'MMD')
    os.makedirs(savedir, exist_ok=True)
    nb_sim = 2000

    # For the left part of figure 2 (H0 being true)
    mmd_values_h0 = np.empty((nb_sim, 1))
    nx_h0 = 50
    ny_h0 = 50

    # For the right part of figure 2 (H1 being true)
    mmd_values_h1 = np.empty((nb_sim, 1))
    nx_h1 = 100
    ny_h1 = 100

    for i in range(nb_sim):

        # Experiment corresponding to the left part of figure 2 - H0 is true
        data_x_h0 = norm.rvs(loc=0, scale=1, size=nx_h0).reshape(-1, 1)
        data_y_h0 = norm.rvs(loc=0, scale=1, size=nx_h0).reshape(-1, 1)
        # RBF kernels with the median heuristic
        kernel = KernelWrapper(
            RBF(length_scale=np.median(np.abs(data_x_h0 - data_y_h0)))
        )
        # Computing MMDu^2
        mmd_values_h0[i] = compute_unbiased_squared_mmd(
            data_x=data_x_h0,
            data_y=data_y_h0,
            kernel=kernel
        )

        # Experiment corresponding to the right part of figure 2 - H1 is true
        data_x_h1 = laplace.rvs(
            loc=0, scale=(math.sqrt(2)/2), size=nx_h1).reshape(-1, 1)
        data_y_h1 = laplace.rvs(loc=0, scale=3, size=ny_h1).reshape(-1, 1)
        # RBF kernels with the median heuristic
        kernel = KernelWrapper(
            RBF(length_scale=np.median(np.abs(data_x_h1 - data_y_h1)))
        )
        # Computing MMDu^2
        mmd_values_h1[i] = compute_unbiased_squared_mmd(
            data_x=data_x_h1,
            data_y=data_y_h1,
            kernel=kernel
        )

    # Save values for the H0 experiment
    df = pd.DataFrame()
    df['MMD'] = mmd_values_h0.reshape((len(mmd_values_h0),))
    csv_filename = os.path.join(savedir, 'MMDu2_values_under_H0.csv')
    df.to_csv(csv_filename, index=False)
    # Save values for the H1 experiment
    df = pd.DataFrame()
    df['MMD'] = mmd_values_h1.reshape((len(mmd_values_h1),))
    csv_filename = os.path.join(savedir, 'MMDu2_values_under_H1.csv')
    df.to_csv(csv_filename, index=False)
    # Save plots for the H0 experiment
    plt.hist(mmd_values_h0, bins='auto', density=True, stacked=True)
    plt.xlabel('MMDu2')
    plt.ylabel('Probability density')
    plt.title('Empirical MMDu2 density under H0')
    plot_filename = os.path.join(savedir, 'MMDu2_histogram_under_H0.png')
    plt.savefig(plot_filename)
    plt.close()
    # Save plots for the H1 experiment
    plt.hist(mmd_values_h1, bins='auto', density=True, stacked=True)
    plt.xlabel('MMDu2')
    plt.ylabel('Probability density')
    plt.title('Empirical MMDu2 density under H1')
    plot_filename = os.path.join(savedir, 'MMDu2_histogram_under_H1.png')
    plt.savefig(plot_filename)
    plt.close()
