import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, gamma
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.mmd import compute_unbiased_mmd


if __name__ == '__main__':

    nb_sim = 5
    mmd_values = np.empty((nb_sim, 1))

    nx = 500
    ny = 510

    for i in range(nb_sim):

        # Two independent N(0, 1) samples
        data_x = norm.rvs(loc=0, scale=1, size=nx).reshape(-1, 1)
        data_y = norm.rvs(loc=0, scale=1, size=ny).reshape(-1, 1)

        # Two RBF kernels with the median heuristic
        kernel = KernelWrapper(RBF(
            # length_scale=np.median(np.abs(data_x - data_y))
            length_scale=np.abs(np.median(data_x) - np.median(data_y))
            )
        )

        mmd_values[i] = compute_unbiased_mmd(data_x=data_x,
                                             data_y=data_y,
                                             kernel=kernel)
        print(mmd_values[i])

    plt.hist(mmd_values, bins='auto', density=True, stacked=True)
    plt.show()

    df = pd.DataFrame()
    df['MMD'] = mmd_values.reshape((len(mmd_values),))

    df.to_csv('mmd_values.csv', index=False)