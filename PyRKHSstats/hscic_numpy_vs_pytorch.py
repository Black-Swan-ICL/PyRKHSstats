import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hscic_checks import run_simulation as run_simulation_numpy
from hscic_pytorch_checks import run_simulation as run_simulation_pytorch


if __name__== '__main__':

    sample_sizes = np.arange(500, 10250, 250)

    time_pytorch = []
    for sample_size in sample_sizes:
        start = time.time()
        run_simulation_pytorch(sample_size=sample_size)
        end = time.time()
        time_taken = end - start
        time_pytorch.append(time_taken)


    time_numpy = []
    for sample_size in sample_sizes:
        start = time.time()
        run_simulation_numpy(sample_size=sample_size)
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
    df.to_csv('comparison_times_hscic.csv', sep=';')

    plt.figure(figsize=(16, 12))
    df.plot()
    plt.legend(loc='best')
    plt.xlabel('Sample size')
    plt.ylabel('Time taken in seconds')
    plt.title('Time taken by the HSCIC simulation for different sample sizes')
    plt.savefig('comparison_times_hscic.png')


