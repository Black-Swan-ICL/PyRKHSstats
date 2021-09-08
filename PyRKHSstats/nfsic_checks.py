import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.fsic import normalised_fsic


def compute_median_distances(x):

    m = x.shape[0]

    distances = np.empty((m, m), dtype=np.double)

    for i in range(m):
        for j in range(m):
            distances[i, j] = np.linalg.norm(x[i] - x[j])

    return np.median(distances)


if __name__ == "__main__":

    N = 1000
    fsic_values = np.empty((N, 1))

    nb_test_points = 10
    dimension = 2

    # Sample test points
    test_points_v = np.random.multivariate_normal(
        mean=[0] * dimension,
        cov=np.identity(dimension),
        size=nb_test_points
    )
    test_points_w = np.random.multivariate_normal(
        mean=[0] * dimension,
        cov=np.identity(dimension),
        size=nb_test_points
    )

    for i in range(N):

        # Generate two independent N_d(0, I_d) samples
        data_x = np.random.multivariate_normal(
            mean=[0]*dimension,
            cov=np.identity(dimension),
            size=100
        )
        data_y = np.random.multivariate_normal(
            mean=[0]*dimension,
            cov=np.identity(dimension),
            size=100
        )

        # Two RBF kernels with the median heuristic
        median_x = compute_median_distances(data_x)
        median_y = compute_median_distances(data_y)
        kernel_k = KernelWrapper(RBF(length_scale=median_x))
        kernel_l = KernelWrapper(RBF(length_scale=median_y))

        fsic = normalised_fsic(data_x=data_x,
                               data_y=data_y,
                               test_points_v=test_points_v,
                               test_points_w=test_points_w,
                               kernel_k=kernel_k,
                               kernel_l=kernel_l,
                               gamma=0.001)

        fsic_values[i] = fsic

    plt.hist(fsic_values, bins='auto', density=True, stacked=True)
    chisquare = chi2(nb_test_points)
    grid = np.linspace(start=0, stop=(max(fsic_values)+1), num=100)
    plt.plot(grid, chisquare.pdf(grid), lw=2)
    plt.show()

    # gamma_parameters = gamma.fit(fsic_values)
    # print(gamma_parameters)
    # with open('gamma_parameters.pickle', 'wb') as f:
    #     pickle.dump(gamma_parameters, f, pickle.HIGHEST_PROTOCOL)

    df = pd.DataFrame()
    df['FSIC'] = fsic_values.reshape((len(fsic_values),))

    df.to_csv('fsic_values.csv', index=False)
