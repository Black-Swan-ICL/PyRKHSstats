"""
This module contains tools to replicate the two plots making up Figure 2 in
'A Kernel Two-Sample Test' A. Gretton, K. M. Borgwardt, M. J. Rasch,
B. Sch\"{o}lkopf and A. Smola (Journal of Machine Learning Research #13, 2012).
"""
import os
import yaml
import time
import math
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple
from datetime import datetime

from scipy.stats import norm, gamma, laplace
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.mmd import compute_unbiased_squared_mmd, \
    compute_biased_squared_mmd


_cfg_name = 'name'
_cfg_nb_obs_x_y = 'sample_sizes_x_y'
_cfg_nb_obs_x = 'nb_observations_x'
_cfg_nb_obs_y = 'nb_observations_y'
_cfg_nb_sim = 'simulation_numbers'


_field_values_x = 'data_x'
_field_values_y = 'data_y'
_field_kernel = 'kernel'
_field_values_mmd = 'MMD'
_field_histogram_xlabel = 'Probability density'


def load_configuration(filepath):

    with open(filepath, "r") as f:
        config_yaml = yaml.safe_load(f)
        f.close()

    dic_cfg = dict()
    dic_cfg[_cfg_name] = config_yaml[_cfg_name]
    dic_cfg[_cfg_nb_sim] = config_yaml[_cfg_nb_sim]
    dic_cfg[_cfg_nb_obs_x_y] = config_yaml[_cfg_nb_obs_x_y]

    return config_yaml


def generate_fig2_left_example(nb_observations_x, nb_observations_y):
    """
    Corresponds to the experiment displayed on the left part of figure 2 in
    'A Kernel Two-Sample Test', A. Gretton, K. M. Borgwardt, M. J. Rasch,
    B. Sch\"{o}lkopf and A. Smola (JMLR #13, 2012).
    """

    data_x = norm.rvs(loc=0, scale=1, size=nb_observations_x).reshape(-1, 1)
    data_y = norm.rvs(loc=0, scale=1, size=nb_observations_y).reshape(-1, 1)

    # RBF kernels with the median heuristic
    kernel = KernelWrapper(
        RBF(length_scale=np.median(np.abs(data_x - data_y)))
    )

    example = dict()
    example[_field_values_x] = data_x
    example[_field_values_y] = data_y
    example[_field_kernel] = kernel

    return example


def generate_fig2_right_example(nb_observations_x, nb_observations_y):
    """
    Corresponds to the experiment displayed on the right part of figure 2 in
    'A Kernel Two-Sample Test', A. Gretton, K. M. Borgwardt, M. J. Rasch,
    B. Sch\"{o}lkopf and A. Smola (JMLR #13, 2012).
    """

    data_x = laplace.rvs(
        loc=0, scale=(math.sqrt(2) / 2), size=nb_observations_x
    ).reshape(-1, 1)
    data_y = laplace.rvs(loc=0, scale=3, size=nb_observations_y).reshape(-1, 1)

    # RBF kernels with the median heuristic
    kernel = KernelWrapper(
        RBF(length_scale=np.median(np.abs(data_x - data_y)))
    )

    example = dict()
    example[_field_values_x] = data_x
    example[_field_values_y] = data_y
    example[_field_kernel] = kernel

    return example


def generate_example_data(example_to_run, nb_observations_x, nb_observations_y,
                          nb_simulations, mmd_function):

    mmd_values = np.zeros((nb_simulations, 1))

    for i in range(nb_simulations):

        example = example_to_run(
            nb_observations_x=nb_observations_x,
            nb_observations_y=nb_observations_y
        )

        mmd_values[i, 0] = mmd_function(
            data_x=example[_field_values_x],
            data_y=example[_field_values_y],
            kernel=example[_field_kernel]
        )['MMD']

    df_example = pd.DataFrame()
    df_example[_field_values_mmd] = mmd_values[:, 0]

    return df_example


def generate_example_plots(data, xlabel, ylabel, title):

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.hist(data, bins='auto', density=True, stacked=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return fig


def run_example(example, id, regime, mmd_function, mmd_function_name,
                nb_observations_x, nb_observations_y, nb_simulations, savedir):

    # Generate data for the example and compute interesting metrics
    start_time = time.time()
    df_example = generate_example_data(
        example_to_run=example,
        nb_observations_x=nb_observations_x,
        nb_observations_y=nb_observations_y,
        nb_simulations=nb_simulations,
        mmd_function=mmd_function
    )
    time_taken = time.time() - start_time

    # Persist the data
    csv_filename = (
        f'MMD_{id}_{mmd_function_name}_{nb_observations_x}_observations_x_' +
        f'{nb_observations_y}_observations_y_{nb_simulations}_simulations.csv'
    )
    csv_filename = os.path.join(savedir, csv_filename)
    df_example.to_csv(csv_filename, index=False)

    # Draw the histogram
    title = (
        f'Empirical {mmd_function_name} density, under {regime} with ' +
        f'{nb_observations_x} observations for X, {nb_observations_y} ' +
        f'observations for Y, over {nb_simulations} simulations'
    )
    fig = generate_example_plots(
        data=df_example[_field_values_mmd].values,
        xlabel=_field_histogram_xlabel,
        ylabel=mmd_function_name,
        title=title
    )
    plot_filename = (
        f'MMD_histogram_{id}_{mmd_function_name}_{nb_observations_x}_' +
        f'observations_x_{nb_observations_y}_observations_y_{nb_simulations}' +
        f'_simulations_run.png'
    )
    plot_filename = os.path.join(savedir, plot_filename)
    fig.savefig(plot_filename)
    plt.close()

    return time_taken


if __name__ == '__main__':

    root_checks_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'checks', 'MMD'
    )
    os.makedirs(root_checks_dir, exist_ok=True)

    parser = argparse.ArgumentParser(
        description='Run checks for the MMD tools implemented.'
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

    simulation_numbers = dic_cfg[_cfg_nb_sim]
    sample_sizes_x_y = dic_cfg[_cfg_nb_obs_x_y]

    examples = dict()
    example1 = namedtuple('id', 'regime')
    example1.id = 'LeftHandFigure2_2012_JMLR_Paper'
    example1.regime = 'H0'
    example2 = namedtuple('id', 'regime')
    example2.id = 'RightHandFigure2_2012_JMLR_Paper'
    example2.regime = 'H1'
    examples[example1] = generate_fig2_left_example
    examples[example2] = generate_fig2_right_example

    # Prepare for the collection of information about the checks
    nb_checks = (
            len(simulation_numbers) * len(sample_sizes_x_y)
    )
    arr_id = np.ndarray(shape=(nb_checks,), dtype=object)
    arr_regime = np.ndarray(shape=(nb_checks,), dtype=object)
    arr_nb_obs_x = np.zeros(nb_checks)
    arr_nb_obs_y = np.zeros(nb_checks)
    arr_nb_sims = np.zeros(nb_checks)
    arr_mmd_func = np.ndarray(shape=(nb_checks,), dtype=object)
    arr_time = np.zeros(nb_checks)

    iter = 0

    # TODO ccould and should be refactored
    # Choose whether to use the biased or unbiased estimator
    mmd_func_dic = dict()
    mmd_func_dic['MMDu^2'] = compute_unbiased_squared_mmd
    mmd_func_dic['MMDb^2'] = compute_biased_squared_mmd
    mmd_func_used = 'MMDu^2'

    # TODO could and should be refactored
    # Choose whether to produce the left plot (example1) or right (example2)
    example_to_run = example2

    for nb_simulations in simulation_numbers:
        for nb_obs_x_y in sample_sizes_x_y:

            time_taken = run_example(
                example=examples[example_to_run],
                id=example_to_run.id,
                regime=example_to_run.regime,
                mmd_function=mmd_func_dic[mmd_func_used],
                mmd_function_name=mmd_func_used,
                nb_observations_x=nb_obs_x_y[_cfg_nb_obs_x],
                nb_observations_y=nb_obs_x_y[_cfg_nb_obs_y],
                nb_simulations=nb_simulations,
                savedir=savedir
            )
            arr_id[iter] = example_to_run.id
            arr_regime[iter] = example_to_run.regime
            arr_nb_obs_x[iter] = nb_obs_x_y[_cfg_nb_obs_x]
            arr_nb_obs_y[iter] = nb_obs_x_y[_cfg_nb_obs_y]
            arr_nb_sims[iter] = nb_simulations
            arr_mmd_func[iter] = mmd_func_used
            arr_time[iter] = time_taken

            iter += 1

    df_summary_checks = pd.DataFrame()
    df_summary_checks['Example ID'] = arr_id
    df_summary_checks['Regime'] = arr_regime
    df_summary_checks['Sample Size for X'] = arr_nb_obs_x
    df_summary_checks['Sample Size for Y'] = arr_nb_obs_y
    df_summary_checks['MMD Function'] = arr_mmd_func
    df_summary_checks['Number Simulations'] = arr_nb_sims
    df_summary_checks['Time Taken in seconds'] = arr_time
    # To timestamp a run
    df_summary_checks['Timestamp Completion Checks'] = str(datetime.now())
    summary_filename = os.path.join(savedir, 'summary_checks.csv')
    df_summary_checks.to_csv(summary_filename, index=False)
