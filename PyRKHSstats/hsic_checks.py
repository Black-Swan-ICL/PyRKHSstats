import os
import copy
import time
import yaml
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple
from datetime import datetime

from scipy.spatial.distance import pdist
from scipy.stats import norm, gamma, multivariate_normal, multivariate_t
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.hsic import perform_hsic_independence_testing, \
    ImplementedHSICScheme


_cfg_name = 'name'
_cfg_schemes = 'schemes'
_cfg_nb_sim = 'simulation_numbers'
_cfg_test_levels = 'test_levels'
_cfg_nb_obs = 'sample_sizes'


_field_values_x = 'data_x'
_field_values_y = 'data_y'
_field_kernel_x = 'kernel_kx'
_field_kernel_y = 'kernel_ky'
_field_values_hsic = 'HSIC'
_field_rejection_threshold = 'Rejection threshold'
_field_null_rejected = 'H0 Rejected'
_field_histogram_xlabel = 'HSIC'
_field_histogram_ylabel = 'Probability density'


# TODO complete
def load_configuration(filepath):

    with open(filepath, "r") as f:
        config_yaml = yaml.safe_load(f)
        f.close()

    dic_cfg = dict()
    dic_cfg[_cfg_schemes] = config_yaml[_cfg_schemes]
    dic_cfg[_cfg_name] = config_yaml[_cfg_name]
    dic_cfg[_cfg_nb_sim] = config_yaml[_cfg_nb_sim]
    dic_cfg[_cfg_test_levels] = config_yaml[_cfg_test_levels]
    dic_cfg[_cfg_nb_obs] = config_yaml[_cfg_nb_obs]

    return dic_cfg


def generate_low_dim_independence_example(nb_observations):

    # Two independent N(0, 1) samples
    data_x = norm.rvs(loc=0, scale=1, size=nb_observations).reshape(-1, 1)
    data_y = norm.rvs(loc=0, scale=1, size=nb_observations).reshape(-1, 1)

    # Two RBF kernels with the median heuristic
    length_scale_kx = np.median(np.abs(pdist(data_x)))
    kernel_kx = KernelWrapper(RBF(length_scale=length_scale_kx))
    length_scale_ky = np.median(np.abs(pdist(data_y)))
    kernel_ky = KernelWrapper(RBF(length_scale=length_scale_ky))

    example = dict()
    example[_field_values_x] = data_x
    example[_field_values_y] = data_y
    example[_field_kernel_x] = kernel_kx
    example[_field_kernel_y] = kernel_ky

    return example


def generate_low_dim_dependence_example(nb_observations):

    # Two independent N(0, 1) samples
    data_x = norm.rvs(loc=0, scale=1, size=nb_observations).reshape(-1, 1)
    data_y = copy.deepcopy(data_x)

    # Two RBF kernels with the median heuristic
    length_scale_kx = np.median(np.abs(pdist(data_x)))
    kernel_kx = KernelWrapper(RBF(length_scale=length_scale_kx))
    kernel_ky = copy.deepcopy(kernel_kx)

    example = dict()
    example[_field_values_x] = data_x
    example[_field_values_y] = data_y
    example[_field_kernel_x] = kernel_kx
    example[_field_kernel_y] = kernel_ky

    return example


def generate_high_dim_independence_example(nb_observations):

    # Two independent N_3(0, I) samples
    data_x = multivariate_normal.rvs(
        mean=[0, 0, 0],
        cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        size=nb_observations
    )
    data_y = multivariate_normal.rvs(
        mean=[0, 0, 0],
        cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        size=nb_observations
    )

    # Two RBF kernels with the median heuristic
    length_scale_kx = np.median(np.abs(pdist(data_x)))
    kernel_kx = KernelWrapper(RBF(length_scale=length_scale_kx))
    length_scale_ky = np.median(np.abs(pdist(data_y)))
    kernel_ky = KernelWrapper(RBF(length_scale=length_scale_ky))

    example = dict()
    example[_field_values_x] = data_x
    example[_field_values_y] = data_y
    example[_field_kernel_x] = kernel_kx
    example[_field_kernel_y] = kernel_ky

    return example


def generate_high_dim_dependence_example(nb_observations):

    data_x = multivariate_t.rvs(
        loc=[0, 0, 0],
        shape=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        df=2.5,
        size=nb_observations
    )
    data_y = copy.deepcopy(data_x)
    data_y += multivariate_t.rvs(
        loc=[0.1, 0.2, 0.3],
        shape=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        df=2.4,
        size=nb_observations
    )

    # Two RBF kernels with the median heuristic
    length_scale_kx = np.median(np.abs(pdist(data_x)))
    kernel_kx = KernelWrapper(RBF(length_scale=length_scale_kx))
    length_scale_ky = np.median(np.abs(pdist(data_y)))
    kernel_ky = KernelWrapper(RBF(length_scale=length_scale_ky))

    example = dict()
    example[_field_values_x] = data_x
    example[_field_values_y] = data_y
    example[_field_kernel_x] = kernel_kx
    example[_field_kernel_y] = kernel_ky

    return example


def generate_example_data(example_to_run, nb_observations, nb_simulations,
                          scheme, test_level):

    values_hsic = np.zeros((nb_simulations, 1))
    values_threshold = np.zeros((nb_simulations, 1))
    values_rejection = np.zeros((nb_simulations, 1))

    for i in range(nb_simulations):

        example = example_to_run(nb_observations=nb_observations)

        hsic_independence_testing_example = perform_hsic_independence_testing(
            data_x=example[_field_values_x],
            data_y=example[_field_values_y],
            kernel_kx=example[_field_kernel_x],
            kernel_ky=example[_field_kernel_y],
            test_level=test_level,
            scheme=scheme
        )

        values_hsic[i, 0] = hsic_independence_testing_example[
            _field_values_hsic
        ]
        values_threshold[i, 0] = hsic_independence_testing_example[
            _field_rejection_threshold
        ]
        values_rejection[i, 0] = (
            1 if values_hsic[i, 0] > values_threshold[i, 0] else 0
        )

    df_example = pd.DataFrame()
    df_example[_field_values_hsic] = values_hsic[:, 0]
    df_example[_field_rejection_threshold] = values_threshold[:, 0]
    df_example[_field_null_rejected] = values_rejection[:, 0]

    return df_example


def generate_example_plots(data, xlabel, ylabel, title):

    fig, ax = plt.subplots(figsize=(16, 10))
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
            f'HSIC_{id}_{scheme.name}_scheme_{nb_observations}_observations' +
            f'_{nb_simulations}_simulations_{test_level}_level.csv'
    )
    csv_filename = os.path.join(savedir, csv_filename)
    df_example.to_csv(csv_filename, index=False)

    # Draw the histogram
    title = (
        f'Empirical HSIC density, {scheme.value} scheme, under {regime} ' +
        f'with {nb_observations} observations, over {nb_simulations} ' +
        'simulations.'
    )
    fig = generate_example_plots(
        data=df_example[_field_values_hsic].values,
        xlabel=_field_histogram_xlabel,
        ylabel=_field_histogram_ylabel,
        title=title
    )
    plot_filename = (
        f'HSIC_histogram_{id}_{scheme.name}_scheme_{nb_observations}_obser' +
        f'vations_{nb_simulations}_simulations_{test_level}_test_level_run.png'
    )
    plot_filename = os.path.join(savedir, plot_filename)
    fig.savefig(plot_filename)
    plt.close()

    return time_taken, nb_h0_rejected


if __name__ == "__main__":

    root_checks_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'checks', 'HSIC'
    )
    os.makedirs(root_checks_dir, exist_ok=True)

    parser = argparse.ArgumentParser(
        description='Run checks for the HSIC tools implemented.'
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Name of the configuration file to use for the run.'
    )
    args = parser.parse_args()

    # Load a pre-defined configuration file that contains the sample sizes,
    # number of simulations etc. to run checks for
    cfg_file = args.config_file
    dic_cfg = load_configuration(
        os.path.join(root_checks_dir, cfg_file)
    )
    # Create folder to house results of current runs
    savedir = os.path.join(root_checks_dir, 'Runs', dic_cfg[_cfg_name])
    os.makedirs(savedir, exist_ok=True)

    schemes = dic_cfg[_cfg_schemes]
    schemes = [ImplementedHSICScheme[elt] for elt in schemes]
    simulation_numbers = dic_cfg[_cfg_nb_sim]
    test_levels = dic_cfg[_cfg_test_levels]
    sample_sizes = dic_cfg[_cfg_nb_obs]

    examples = dict()
    example1 = namedtuple('id', 'regime')
    example1.id = 'LowDimIndependence'
    example1.regime = 'H0'
    examples[example1] = generate_low_dim_independence_example
    example2 = namedtuple('id', 'regime')
    example2.id = 'LowDimDependence'
    example2.regime = 'H1'
    examples[example2] = generate_low_dim_dependence_example
    example3 = namedtuple('id', 'regime')
    example3.id = 'HighDimInDependence'
    example3.regime = 'H0'
    examples[example3] = generate_high_dim_independence_example
    example4 = namedtuple('id', 'regime')
    example4.id = 'HighDimDependence'
    example4.regime = 'H1'
    examples[example4] = generate_high_dim_dependence_example

    # Prepare for the collection of information about the checks
    nb_checks = (
            len(schemes) * len(simulation_numbers) * len(sample_sizes) *
            len(test_levels) * len(examples.keys())
    )
    arr_id = np.ndarray(shape=(nb_checks,), dtype=object)
    arr_regime = np.ndarray(shape=(nb_checks,), dtype=object)
    arr_nb_obs = np.zeros(nb_checks)
    arr_level = np.zeros(nb_checks)
    arr_nb_sims = np.zeros(nb_checks)
    arr_scheme = np.ndarray(shape=(nb_checks,), dtype=object)
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
    # To timestamp a run
    df_summary_checks['Timestamp Completion Checks'] = str(datetime.now())
    summary_filename = os.path.join(savedir, 'summary_checks.csv')
    df_summary_checks.to_csv(summary_filename, index=False)
