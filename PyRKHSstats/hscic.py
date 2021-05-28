import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.gaussian_process.kernels import RBF

from PyRKHSstats.kernel_wrapper import KernelWrapper


def vec_k_Z(z, data_z, kernel_z):

    n = len(data_z)  # TODO you sure about that ?
    res = np.zeros((n, 1))
    for i in range(n):

        res[i, 0] = kernel_z.evaluate(data_z[i], z)

    return res


# TODO replace the np.matmul by the '@' operator
def hscic(z, data_x, data_y, data_z, kernel_x, kernel_y, kernel_z,
          regularisation_cst):

    # Compute the kernelised Gram matrices
    mat_K_X = kernel_x.compute_kernelised_gram_matrix(data_x)
    mat_K_Y = kernel_y.compute_kernelised_gram_matrix(data_y)
    mat_K_Z = kernel_z.compute_kernelised_gram_matrix(data_z)

    n = mat_K_X.shape[0]

    mat_W = np.linalg.inv(mat_K_Z + n * regularisation_cst * np.identity(n))
    vec_k_Z_in_z = vec_k_Z(z, data_z, kernel_z)

    term_1 = np.matmul(
        np.transpose(vec_k_Z_in_z),
        np.matmul(
            mat_W,
            np.matmul(
                np.multiply(mat_K_X, mat_K_Y),
                np.matmul(
                    np.transpose(mat_W),
                    vec_k_Z_in_z
                )
            )
        )
    )

    term_2 = np.matmul(
        np.transpose(vec_k_Z_in_z),
        np.matmul(
            mat_W,
            np.multiply(
                np.matmul(
                    mat_K_X,
                    np.matmul(
                        np.transpose(mat_W),
                        vec_k_Z_in_z
                    )
                ),
                np.matmul(
                    mat_K_Y,
                    np.matmul(
                        np.transpose(mat_W),
                        vec_k_Z_in_z
                    )
                )
            )
        )
    )

    term_3 = np.matmul(
        np.matmul(
            np.transpose(vec_k_Z_in_z),
            np.matmul(
                mat_W,
                np.matmul(
                    mat_K_X,
                    np.matmul(
                        np.transpose(mat_W),
                        vec_k_Z_in_z
                    )
                )
            )
        ),
        np.matmul(
            np.transpose(vec_k_Z_in_z),
            np.matmul(
                mat_W,
                np.matmul(
                    mat_K_Y,
                    np.matmul(
                        np.transpose(mat_W),
                        vec_k_Z_in_z
                    )
                )
            )
        ),
    )

    res = term_1 - 2 * term_2 + term_3

    return res


if __name__ == '__main__':

    # Reproducing the figures from 'A Measure-Theoretic Approach to Kernel
    # Conditional Mean Embedding' (J. Park, K.Muandet arXiv:2002.03689v8).

    sample_size = 500
    regularisation_cst = 0.01
    length_scale = 0.1
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
    # Figure 3 (a)
    plt.figure(figsize=(10, 8))
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
    hscic_value = hscic(z=0.,
                        data_x=X,
                        data_y=Y_noise,
                        data_z=Z,
                        kernel_x=kernel_x,
                        kernel_y=kernel_y,
                        kernel_z=kernel_z,
                        regularisation_cst=regularisation_cst)
    print(hscic_value)
    hscic_value = hscic(z=0.,
                        data_x=X,
                        data_y=Y_dep_add,
                        data_z=Z,
                        kernel_x=kernel_x,
                        kernel_y=kernel_y,
                        kernel_z=kernel_z,
                        regularisation_cst=regularisation_cst)
    print(hscic_value)
    hscic_value = hscic(z=0.,
                        data_x=X,
                        data_y=Yprime_dep_add,
                        data_z=Z,
                        kernel_x=kernel_x,
                        kernel_y=kernel_y,
                        kernel_z=kernel_z,
                        regularisation_cst=regularisation_cst)
    print(hscic_value)
