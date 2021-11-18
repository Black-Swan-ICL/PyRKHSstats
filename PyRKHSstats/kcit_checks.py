import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple

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


# Low-dimensional settings example, under which CI holds true
def generate_low_dim_ci_example(nb_observations):

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


# Low-dimensional settings example, under which CI does NOT hold true
def generate_low_dim_not_ci_example(nb_observations):

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


def generate_example_data(example_to_run, nb_observations, nb_simulations,
                          scheme, test_level):

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


def run_example(example, id, regime, nb_observations, nb_simulations, scheme,
                test_level, savedir):

    # Generate data for the example and compute interesting metrics
    start_time = time.time()
    df_example = generate_example_data(
        example_to_run=example,
        nb_observations=nb_observations,
        nb_simulations=nb_simulations,
        scheme=scheme,
        test_level=test_level
    )
    time_taken = time.time() - start_time
    nb_h0_rejected = np.sum(df_example[_field_null_rejected].values)
    # Persist the data
    csv_filename = (
            f'KCIT_{id}_{nb_observations}_observations' +
            f'_{nb_simulations}_simulations_{test_level}_level.csv'
    )
    csv_filename = os.path.join(savedir, csv_filename)
    df_example.to_csv(csv_filename, index=False)

    # Draw the histogram
    title = (
        f'Empirical TCI density under {regime} with ' +
        f'{nb_observations} observations, over {nb_simulations} simulations.'
    )
    fig = generate_example_plots(
        data=df_example[_field_values_tci].values,
        xlabel=_field_histogram_xlabel,
        ylabel=_field_histogram_ylabel,
        title=title
    )
    plot_filename = (
        f'TCI_histogram_{id}_{nb_observations}_observations_' +
        f'{nb_simulations}_simulations_{test_level}_test_level_run.png'
    )
    plot_filename = os.path.join(savedir, plot_filename)
    fig.savefig(plot_filename)
    plt.close()

    return time_taken, nb_h0_rejected


if __name__ == '__main__':

    savedir = os.path.join('checks', 'KCIT')
    os.makedirs(savedir, exist_ok=True)

    schemes = [ImplementedKCITSchemes.GAMMA]
    simulation_numbers = [100, 1000]
    test_levels = [0.05, 0.01]
    sample_sizes = [100, 200]

    examples = dict()
    example1 = namedtuple('id', 'regime')
    example1.id = 'Example1'
    example1.regime = 'H0'
    example2 = namedtuple('id', 'regime')
    example2.id = 'Example2'
    example2.regime = 'H1'
    examples[example1] = generate_low_dim_ci_example
    examples[example2] = generate_low_dim_not_ci_example

    # Prepare for the collection of information about the checks
    nb_checks = (
            len(schemes) * len(simulation_numbers) * len(sample_sizes) *
            len(test_levels) * len(examples.keys())
    )
    arr_id = np.ndarray(shape=(nb_checks,), dtype=object)
    arr_regime =  np.ndarray(shape=(nb_checks,), dtype=object)
    arr_nb_obs = np.zeros(nb_checks)
    arr_level = np.zeros(nb_checks)
    arr_nb_sims = np.zeros(nb_checks)
    arr_scheme =  np.ndarray(shape=(nb_checks,), dtype=object)
    arr_h0_reject = np.zeros(nb_checks)
    arr_time = np.zeros(nb_checks)

    iter = 0

    for scheme in schemes:
        for nb_simulations in simulation_numbers:
            for nb_observations in sample_sizes:
                for test_level in test_levels:
                    for example_setup, example in examples.items():

                        time_taken, nb_h0_rejected = run_example(
                            example=example,
                            id=example_setup.id,
                            regime=example_setup.regime,
                            nb_observations=nb_observations,
                            nb_simulations=nb_simulations,
                            scheme=scheme,
                            test_level=test_level,
                            savedir=savedir
                        )

                        arr_id[iter] = example_setup.id
                        arr_regime[iter] = example_setup.regime
                        arr_nb_obs[iter] = nb_observations
                        arr_level[iter] = test_level
                        arr_nb_sims[iter] = nb_simulations
                        arr_scheme[iter] = scheme
                        arr_h0_reject[iter] = nb_h0_rejected
                        arr_time[iter] = time_taken

                        iter += 1

    df_summary_checks = pd.DataFrame()
    df_summary_checks['Example ID'] = arr_id
    df_summary_checks['Regime'] = arr_regime
    df_summary_checks['Sample Size'] = arr_nb_obs
    df_summary_checks['Test Level'] = arr_level
    df_summary_checks['Number Simulations'] = arr_nb_sims
    df_summary_checks['Scheme'] = arr_scheme
    df_summary_checks['Proportion H0 Rejected'] = arr_h0_reject / arr_nb_sims
    df_summary_checks['Time Taken in seconds'] = arr_time
    summary_filename = os.path.join(savedir, 'summary_checks.csv')
    df_summary_checks.to_csv(summary_filename, index=False)
