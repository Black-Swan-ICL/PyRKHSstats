import os

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hscic_neurips_paper_replication import run_simulation as \
    run_simulation_numpy
from hscic_pytorch_neurips_paper_replication import run_simulation as \
    run_simulation_pytorch


if __name__ == '__main__':

    root_checks_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'checks', 'HSCIC'
    )
    os.makedirs(root_checks_dir, exist_ok=True)

    sample_sizes = np.arange(500, 10250, 250)

    time_pytorch = []
    for sample_size in sample_sizes:
        start = time.time()
        run_simulation_pytorch(
            savedir=root_checks_dir,
            sample_size=sample_size
        )
        end = time.time()
        time_taken = end - start
        time_pytorch.append(time_taken)


    time_numpy = []
    for sample_size in sample_sizes:
        start = time.time()
        run_simulation_numpy(
            savedir=root_checks_dir,
            sample_size=sample_size
        )
        end = time.time()
        time_taken = end - start
        time_numpy.append(time_taken)

    df = pd.DataFrame(
        data={
            'TimeWithNumpy': time_numpy,
            'TimeWithPyTorch': time_pytorch
        },
        index=sample_sizes
    )
    csv_filename = os.path.join(root_checks_dir, 'comparison_times_hscic.csv')
    df.to_csv(csv_filename, sep=';')

    plt.figure(figsize=(16, 12))
    df.plot()
    plt.legend(loc='best')
    plt.xlabel('Sample size')
    plt.ylabel('Time taken in seconds')
    plt.title('Time taken by the HSCIC simulation for different sample sizes')
    plot_filename = os.path.join(root_checks_dir, 'comparison_times_hscic.png')
    plt.savefig(plot_filename)


