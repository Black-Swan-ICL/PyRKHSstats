import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, gamma
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.hsic import biased_hsic


if __name__ == "__main__":

    N = 10000
    hsic_values = np.empty((N, 1))

    for i in range(N):

        # Two independent N(0, 1) samples
        data_x = norm.rvs(loc=0, scale=1, size=1000).reshape(-1, 1)
        data_y = norm.rvs(loc=0, scale=1, size=1000).reshape(-1, 1)

        # Two RBF kernels with the median heuristic
        kernel_k = KernelWrapper(RBF(
            length_scale=np.median(np.abs(data_x - data_y))))
        kernel_l = KernelWrapper(RBF(
            length_scale=np.median(np.abs(data_x - data_y))))

        dic = biased_hsic(data_x=data_x,
                          data_y=data_y,
                          kernel_k=kernel_k,
                          kernel_l=kernel_l)
        hsic = dic['HSIC']

        hsic_values[i] = hsic

    plt.hist(hsic_values, bins='auto', density=True, stacked=True)
    plt.show()

    gamma_parameters = gamma.fit(hsic_values)
    print(gamma_parameters)
    with open('gamma_parameters.pickle', 'wb') as f:
        pickle.dump(gamma_parameters, f, pickle.HIGHEST_PROTOCOL)

    df = pd.DataFrame()
    df['HSIC'] = hsic_values.reshape((len(hsic_values),))

    df.to_csv('hsic_values.csv', index=False)
