import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.stats import norm, gamma, laplace
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.kcit import perform_kcit, ImplementedKCITSchemes


def generate_low_dimensional_ci_example(nb_observations):

    # Low-dimensional where CI holds true
    data_z = laplace.rvs(loc=0,
                         scale=1,
                         size=nb_observations).reshape(-1, 1)
    data_x = 2 * data_z + laplace.rvs(loc=0,
                                      scale=1,
                                      size=nb_observations).reshape(-1, 1)
    data_y = data_z + laplace.rvs(loc=0,
                                  scale=1,
                                  size=nb_observations).reshape(-1, 1)

    # Kernels to use
    length_scale_kx = np.median(np.abs(pdist(data_x)))
    kernel_kx = KernelWrapper(RBF(length_scale=length_scale_kx))
    length_scale_ky = np.median(np.abs(pdist(data_y)))
    kernel_ky = KernelWrapper(RBF(length_scale=length_scale_ky))
    length_scale_kz = np.median(np.abs(pdist(data_z)))
    kernel_kz = KernelWrapper(RBF(length_scale=length_scale_kz))

    # Hyperparameter - arbitrary value
    epsilon = 0.001

    example = dict()
    example['data_x'] = data_x
    example['data_y'] = data_y
    example['data_z'] = data_z
    example['kernel_kx'] = kernel_kx
    example['kernel_ky'] = kernel_ky
    example['kernel_kz'] = kernel_kz
    example['epsilon'] = epsilon

    return example


def generate_low_dimensional_not_ci_example(nb_observations):

    # Low-dimensional where CI does not hold
    data_x = laplace.rvs(loc=0.5,
                         scale=1,
                         size=nb_observations).reshape(-1, 1)
    data_y = laplace.rvs(loc=0,
                         scale=1,
                         size=nb_observations).reshape(-1, 1)
    data_z = data_x + data_y + laplace.rvs(loc=0,
                                           scale=1,
                                           size=nb_observations).reshape(-1, 1)

    # Kernels to use
    length_scale_kx = np.median(np.abs(pdist(data_x)))
    kernel_kx = KernelWrapper(RBF(length_scale=length_scale_kx))
    length_scale_ky = np.median(np.abs(pdist(data_y)))
    kernel_ky = KernelWrapper(RBF(length_scale=length_scale_ky))
    length_scale_kz = np.median(np.abs(pdist(data_z)))
    kernel_kz = KernelWrapper(RBF(length_scale=length_scale_kz))

    # Hyperparameter - arbitrary value
    epsilon = 0.001

    example = dict()
    example['data_x'] = data_x
    example['data_y'] = data_y
    example['data_z'] = data_z
    example['kernel_kx'] = kernel_kx
    example['kernel_ky'] = kernel_ky
    example['kernel_kz'] = kernel_kz
    example['epsilon'] = epsilon

    return example


if __name__ == '__main__':

    savedir = os.path.join('checks', 'KCIT')
    os.makedirs(savedir, exist_ok=True)

    nb_sim = 100

    test_level = 0.05
    N = 100

    # Example I : low-dimensioanl settings, CI holds true
    values_tci = np.zeros((nb_sim, 1))
    values_threshold = np.zeros((nb_sim, 1))
    values_rejection = np.zeros((nb_sim, 1))

    for i in range(nb_sim):

        example1 = generate_low_dimensional_ci_example(nb_observations=N)

        kcit_example1 = perform_kcit(
            data_x=example1['data_x'],
            data_y=example1['data_y'],
            data_z=example1['data_z'],
            kernel_kx=example1['kernel_kx'],
            kernel_ky=example1['kernel_ky'],
            kernel_kz=example1['kernel_kz'],
            epsilon=example1['epsilon'],
            test_level=test_level,
            scheme=ImplementedKCITSchemes.GAMMA
        )

        values_tci[i, 0] = kcit_example1['TCI']
        values_threshold[i, 0] = kcit_example1['Rejection threshold']
        values_rejection[i, 0] = (
            1 if values_tci[i, 0] > values_threshold[i, 0] else 0
        )

    df_example1 = pd.DataFrame()
    df_example1['TCI'] = values_tci[:, 0]
    df_example1['Rejection Threshold'] = values_threshold[:, 0]
    df_example1['H0 Rejected'] = values_rejection[:, 0]
    csv_filename = os.path.join(savedir, 'KCIT_example1.csv')
    df_example1.to_csv(csv_filename, index=False)

    plt.hist(values_tci[:, 0], bins='auto', density=True, stacked=True)
    plt.xlabel('TCI')
    plt.ylabel('Probability density')
    plt.title('Empirical TCI density under H0')
    plot_filename = os.path.join(savedir, 'TCI_histogram_example1.png')
    plt.savefig(plot_filename)
    plt.close()

    # Example II : low-dimensional, CI does not hold
    values_tci = np.zeros((nb_sim, 1))
    values_threshold = np.zeros((nb_sim, 1))
    values_rejection = np.zeros((nb_sim, 1))

    for i in range(nb_sim):

        example2 = generate_low_dimensional_ci_example(nb_observations=N)

        kcit_example2 = perform_kcit(
            data_x=example2['data_x'],
            data_y=example2['data_y'],
            data_z=example2['data_z'],
            kernel_kx=example2['kernel_kx'],
            kernel_ky=example2['kernel_ky'],
            kernel_kz=example2['kernel_kz'],
            epsilon=example2['epsilon'],
            test_level=test_level,
            scheme=ImplementedKCITSchemes.GAMMA
        )

        values_tci[i, 0] = kcit_example2['TCI']
        values_threshold[i, 0] = kcit_example2['Rejection threshold']
        values_rejection[i, 0] = (
            1 if values_tci[i, 0] > values_threshold[i, 0] else 0
        )

    df_example2 = pd.DataFrame()
    df_example2['TCI'] = values_tci[:, 0]
    df_example2['Rejection Threshold'] = values_threshold[:, 0]
    df_example2['H0 Rejected'] = values_rejection[:, 0]
    csv_filename = os.path.join(savedir, 'KCIT_example2.csv')
    df_example2.to_csv(csv_filename, index=False)

    plt.hist(values_tci[:, 0], bins='auto', density=True, stacked=True)
    plt.xlabel('TCI')
    plt.ylabel('Probability density')
    plt.title('Empirical TCI density under H1')
    plot_filename = os.path.join(savedir, 'TCI_histogram_example2.png')
    plt.savefig(plot_filename)
    plt.close()
