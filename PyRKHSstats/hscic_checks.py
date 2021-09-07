import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper
from PyRKHSstats.utilities import timer
from PyRKHSstats.hscic import compute_mat_W, compute_vec_k_Z_in_z, \
    compute_hscic


def run_simulation(sample_size=500, produce_plots=False):
    # to be able to replicate the figures in the article, need to divide by
    # the sample size
    regularisation_cst = 0.01 / sample_size
    length_scale = 0.1 ** (-0.5)
    kernel_x = KernelWrapper(RBF(length_scale=length_scale))
    kernel_y = KernelWrapper(RBF(length_scale=length_scale))
    kernel_z = KernelWrapper(RBF(length_scale=length_scale))

    # Figure 3 - Additive noise model
    Z = norm.rvs(loc=0, scale=1, size=sample_size)
    NX = 0.3 * norm.rvs(loc=0, scale=1, size=sample_size)
    X = np.exp(-0.5 * np.power(Z, 2)) * np.sin(2 * Z) + NX
    NY = 0.3 * norm.rvs(loc=0, scale=1, size=sample_size)
    Y_noise = NY
    Y_dep_add = 1.2 * X
    Yprime_dep_add = 1.4 * X
    # Reshaping for the Gram matrix computation
    Z = Z.reshape(-1, 1)
    X = X.reshape(-1, 1)
    Y_noise = Y_noise.reshape(-1, 1)
    Y_dep_add = Y_dep_add.reshape(-1, 1)
    Yprime_dep_add = Yprime_dep_add.reshape(-1, 1)

    if produce_plots:
        # Figure 3 (a)
        plt.figure(figsize=(20, 15))
        plt.scatter(Z, X, c='blue', marker='o', label='X')
        plt.scatter(Z, Y_noise, c='orange', marker='*', label='Y_noise')
        plt.scatter(Z, Y_dep_add, c='green', marker='x', label='Y_dep_add')
        plt.scatter(Z, Yprime_dep_add, c='red', marker='D', label="Y'_dep_add")
        plt.legend(loc='best')
        plt.xlabel('z')
        plt.ylabel('x, y')
        plt.title('Simulated Data - Additive Noise')
        plt.savefig('Figure3a.png')
        plt.close()

    # Figure 3 (b)
    eval_points = np.arange(-10, 10, 0.05)

    mat_K_X = kernel_x.compute_kernelised_gram_matrix(X)
    mat_K_Y = kernel_y.compute_kernelised_gram_matrix(Y_noise)
    mat_K_Z = kernel_z.compute_kernelised_gram_matrix(Z)
    n = mat_K_X.shape[0]
    mat_W = compute_mat_W(mat_K_Z=mat_K_Z,
                          regularisation_constant=regularisation_cst)

    def func_vec_k_Z(z):
        return compute_vec_k_Z_in_z(z, Z, kernel_z)

    # @timer
    def compute_hscic_values(eval_points, mat_K_X, mat_K_Y, hadamard_K_X_K_Y,
                             mat_W, func_vec_k_Z):

        hscic_values = np.zeros_like(eval_points)
        hscic_values[:] = np.nan
        for i in range(len(eval_points)):
            hscic_values[i] = compute_hscic(z=eval_points[i],
                                            mat_K_X=mat_K_X,
                                            mat_K_Y=mat_K_Y,
                                            hadamard_K_X_K_Y=hadamard_K_X_K_Y,
                                            mat_W=mat_W,
                                            func_vec_k_Z=func_vec_k_Z)

        return hscic_values

    # First example
    hadamard_K_X_K_Y = np.multiply(mat_K_X, mat_K_Y)
    hscic_values_1 = compute_hscic_values(
        eval_points=eval_points,
        mat_K_X=mat_K_X,
        mat_K_Y=mat_K_Y,
        hadamard_K_X_K_Y=hadamard_K_X_K_Y,
        mat_W=mat_W,
        func_vec_k_Z=func_vec_k_Z
    )
    # Second example
    mat_K_Y_dep_add = kernel_y.compute_kernelised_gram_matrix(Y_dep_add)
    hadamard_K_X_K_Y_dep_add = np.multiply(mat_K_X, mat_K_Y_dep_add)
    hscic_values_2 = compute_hscic_values(
        eval_points=eval_points,
        mat_K_X=mat_K_X,
        mat_K_Y=mat_K_Y_dep_add,
        hadamard_K_X_K_Y=hadamard_K_X_K_Y_dep_add,
        mat_W=mat_W,
        func_vec_k_Z=func_vec_k_Z
    )
    # Third example
    mat_K_Yprime_dep_add = kernel_y.compute_kernelised_gram_matrix(
        Yprime_dep_add
    )
    hadamard_K_X_K_Yprime_dep_add = np.multiply(mat_K_X, mat_K_Yprime_dep_add)
    hscic_values_3 = compute_hscic_values(
        eval_points=eval_points,
        mat_K_X=mat_K_X,
        mat_K_Y=mat_K_Yprime_dep_add,
        hadamard_K_X_K_Y=hadamard_K_X_K_Yprime_dep_add,
        mat_W=mat_W,
        func_vec_k_Z=func_vec_k_Z
    )

    if produce_plots:
        plt.figure(figsize=(20, 15))
        plt.scatter(
            eval_points,
            hscic_values_1,
            c='orange',
            label='HSCIC(X, Y_noise | Z)'
        )
        plt.scatter(
            eval_points,
            hscic_values_2,
            c='green',
            label='HSCIC(X, Y_dep_add | Z)'
        )
        plt.scatter(
            eval_points,
            hscic_values_3,
            c='red',
            label="HSCIC(X, Y'_dep_add | Z)"
        )
        plt.legend(loc='best')
        plt.xlabel('z')
        plt.title('HSCIC values')
        plt.savefig('Figure3b.png')
        plt.close()


if __name__ == '__main__':

    # Reproducing the figures from 'A Measure-Theoretic Approach to Kernel
    # Conditional Mean Embedding' (J. Park, K.Muandet arXiv:2002.03689v8).
    np.random.seed(22)
    run_simulation(sample_size=500, produce_plots=True)
