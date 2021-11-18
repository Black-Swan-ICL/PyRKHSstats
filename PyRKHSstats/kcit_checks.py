import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist
from scipy.stats import norm, gamma, laplace
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.kcit import perform_kcit, ImplementedKCITSchemes


_field_values_x = 'data_x'
_field_values_y = 'data_y'
_field_values_z = 'data_z'
_field_kernel_x = 'kernel_kx'
_field_kernel_y = 'kernel_ky'
_field_kernel_z = 'kernel_kz'
_field_regularisation_constant = 'epsilon'
_field_values_tci = 'TCI'
_field_rejection_threshold = 'Rejection threshold'
_field_null_rejected = 'H0 Rejected'
_field_histogram_xlabel = 'TCI'
_field_histogram_ylabel = 'Probability density'


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
    example[_field_values_x] = data_x
    example[_field_values_y] = data_y
    example[_field_values_z] = data_z
    example[_field_kernel_x] = kernel_kx
    example[_field_kernel_y] = kernel_ky
    example[_field_kernel_z] = kernel_kz
    example[_field_regularisation_constant] = epsilon

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
    example[_field_values_x] = data_x
    example[_field_values_y] = data_y
    example[_field_values_z] = data_z
    example[_field_kernel_x] = kernel_kx
    example[_field_kernel_y] = kernel_ky
    example[_field_kernel_z] = kernel_kz
    example[_field_regularisation_constant] = epsilon

    return example


def generate_example_data(example_to_run, nb_observations, nb_simulations, scheme,
                          test_level):

    values_tci = np.zeros((nb_simulations, 1))
    values_threshold = np.zeros((nb_simulations, 1))
    values_rejection = np.zeros((nb_simulations, 1))

    for i in range(nb_simulations):

        example = example_to_run(nb_observations=nb_observations)

        kcit_example = perform_kcit(
            data_x=example[_field_values_x],
            data_y=example[_field_values_y],
            data_z=example[_field_values_z],
            kernel_kx=example[_field_kernel_x],
            kernel_ky=example[_field_kernel_y],
            kernel_kz=example[_field_kernel_z],
            epsilon=example[_field_regularisation_constant],
            test_level=test_level,
            scheme=scheme
        )

        values_tci[i, 0] = kcit_example[_field_values_tci]
        values_threshold[i, 0] = kcit_example[_field_rejection_threshold]
        values_rejection[i, 0] = (
            1 if values_tci[i, 0] > values_threshold[i, 0] else 0
        )

    df_example = pd.DataFrame()
    df_example[_field_values_tci] = values_tci[:, 0]
    df_example[_field_rejection_threshold] = values_threshold[:, 0]
    df_example[_field_null_rejected] = values_rejection[:, 0]

    return df_example


def generate_example_plots(data, xlabel, ylabel, title):

    fig, ax = plt.subplots()
    ax.hist(data, bins='auto', density=True, stacked=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig


if __name__ == '__main__':

    savedir = os.path.join('checks', 'KCIT')
    os.makedirs(savedir, exist_ok=True)

    nb_sim = 1000

    test_level = 0.01
    N = 100

    start_time = time.time()

    # Example I : low-dimensional settings, CI holds true
    df_example1 = generate_example_data(
        example_to_run=generate_low_dimensional_ci_example,
        nb_observations=N,
        nb_simulations=nb_sim,
        scheme=ImplementedKCITSchemes.GAMMA,
        test_level=test_level
    )
    csv_filename = os.path.join(
        savedir,
        (
                f'KCIT_example1_{N}_observations_{nb_sim}_simulations_' +
                f'{test_level}_level.csv'
        )
    )
    df_example1.to_csv(csv_filename, index=False)

    title = f'Empirical TCI density under H0 with {N} observations, over ' \
            f'{nb_sim} simulations.'
    fig = generate_example_plots(
        data=df_example1[_field_values_tci].values,
        xlabel=_field_histogram_xlabel,
        ylabel=_field_histogram_ylabel,
        title=title
    )
    plot_filename = os.path.join(
        savedir,
        (
            f'TCI_histogram_example1_{N}_observations_{nb_sim}_simulations_' +
            f'{test_level}_test_level_run.png'
        )
    )
    fig.savefig(plot_filename)
    plt.close()

    # Example II : low-dimensional, CI does not hold
    df_example2 = generate_example_data(
        example_to_run=generate_low_dimensional_not_ci_example,
        nb_observations=N,
        nb_simulations=nb_sim,
        scheme=ImplementedKCITSchemes.GAMMA,
        test_level=test_level
    )
    csv_filename = os.path.join(
        savedir,
        (
                f'KCIT_example2_{N}_observations_{nb_sim}_simulations_' +
                f'{test_level}_level.csv'
        )
    )
    df_example2.to_csv(csv_filename, index=False)

    title = f'Empirical TCI density under H1 with {N} observations, over ' \
            f'{nb_sim} simulations.'
    fig = generate_example_plots(
        data=df_example2[_field_values_tci].values,
        xlabel=_field_histogram_xlabel,
        ylabel=_field_histogram_ylabel,
        title=title
    )
    plot_filename = os.path.join(
        savedir,
        (
            f'TCI_histogram_example2_{N}_observations_{nb_sim}_simulations_' +
            f'{test_level}_test_level_run.png'
        )
    )
    fig.savefig(plot_filename)
    plt.close()

    end_time = time.time()

    print(end_time - start_time)
